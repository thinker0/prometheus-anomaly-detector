import asyncio
import logging
import os
import threading
import time
from datetime import datetime
from functools import partial
from multiprocessing import Queue
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from queue import Empty as EmptyQueueException

import schedule
import tornado.ioloop
import tornado.web
import tornado.web
from prometheus_api_client import PrometheusConnect, Metric
from prometheus_client import Gauge, generate_latest, REGISTRY
from tornado import httpserver
from tornado.options import parse_command_line

import model
import model_fourier
import model_lstm
import model_sarima
from configuration import Configuration

# Set up logging
_LOGGER = logging.getLogger(__name__)

# list of metrics to be scraped
METRICS_LIST = Configuration.metrics_list

# list of ModelPredictor Objects shared between processes
PREDICTOR_MODEL_LIST = list()

# model list
MODEL_LIST = {
    "prophet": model,
    "fourier": model_fourier,
    "lstm": model_lstm,
    "sarima": model_sarima,
}

# Prometheus Connect Object
pc = PrometheusConnect(
    url=Configuration.prometheus_url,
    headers=Configuration.prom_connect_headers,
    disable_ssl=False,
)


def remove_ignore_metrics(unique_metric):
    """Remove metrics from the gauge dict which are not in the config."""
    for label_name in Configuration.remove_metric_labels:
        unique_metric["metric"].pop(label_name, None)


# Initialize the predictor models
for metric in METRICS_LIST:
    # Initialize a predictor for all metrics first
    metric_init = pc.get_current_metric_value(metric_name=metric)

    for unique_metric in metric_init:
        remove_ignore_metrics(unique_metric)
        PREDICTOR_MODEL_LIST.append(
            MODEL_LIST[Configuration.model_type.lower()].MetricPredictor(
                unique_metric,
                rolling_data_window_size=Configuration.rolling_training_window_size,
            )
        )

# A gauge set for the predicted values
GAUGE_DICT = dict()
for predictor in PREDICTOR_MODEL_LIST:
    unique_metric = predictor.metric
    label_list = list(unique_metric.label_config.keys())
    label_list.append("value_type")
    if unique_metric.metric_name not in GAUGE_DICT:
        GAUGE_DICT[unique_metric.metric_name] = Gauge(
            unique_metric.metric_name + "_" + predictor.model_name.lower(),
            predictor.model_description,
            label_list,
        )


# Main Handler for the web app
class MainApplication(tornado.web.Application):
    data_queue = None
    running = True

    """Tornado web application."""

    def __init__(self, queue):
        self.data_queue = queue
        """Initialize the tornado web app."""
        _LOGGER.info("Initializing Tornado Web App")
        handlers = [
            (r"/health", HealthHandler, dict()),
            (r"/quitquitquit", LifecycleHandler, dict(app=self)),
            (r"/abortabortabort", LifecycleHandler, dict(app=self)),
            (r"/metrics", MainHandler, dict(data_queue=queue)),
            (r"/", MainHandler, dict(data_queue=queue)),
        ]
        settings = dict(
            debug=True,
        )
        super(MainApplication, self).__init__(handlers, **settings)

    @staticmethod
    def train_individual_model(predictor_model, initial_run):
        metric_to_predict = predictor_model.metric
        # Get the start time for the metric data
        data_start_time = datetime.now() - Configuration.metric_chunk_size
        if initial_run:
            data_start_time = (
                    datetime.now() - Configuration.rolling_training_window_size
            )

        # Download new metric data from prometheus
        new_metric_data = pc.get_metric_range_data(
            metric_name=metric_to_predict.metric_name,
            label_config=metric_to_predict.label_config,
            start_time=data_start_time,
            end_time=datetime.now(),
        )[0]
        # Remove the metrics which are not in the config
        remove_ignore_metrics(new_metric_data)
        # Train the new model
        start_time = datetime.now()
        predictor_model.train(
            new_metric_data, Configuration.retraining_interval_minutes)

        _LOGGER.info(
            "Total Training time taken = %s, for metric: %s %s",
            str(datetime.now() - start_time),
            metric_to_predict.metric_name,
            metric_to_predict.label_config,
        )
        return predictor_model

    def train_model(self, initial_run=False, data_queue=None):
        """Train the machine learning model."""
        global PREDICTOR_MODEL_LIST
        parallelism = min(Configuration.parallelism, cpu_count())
        _LOGGER.info(f"Training models using ProcessPool of size:{parallelism}")
        training_partial = partial(self.train_individual_model, initial_run=initial_run)
        with ThreadPool(parallelism) as p:
            result = p.map(training_partial, PREDICTOR_MODEL_LIST)
        PREDICTOR_MODEL_LIST = result
        data_queue.put(PREDICTOR_MODEL_LIST)

    def loop_forever(self):
        """Run the main loop forever."""
        _LOGGER.info("Starting the main loop")
        while self.running:
            # jobs = schedule.get_jobs()
            # _LOGGER.info("Starting scheduler for training %s", len(jobs))
            schedule.run_pending()
            time.sleep(1)

    def stop(self):
        """Stop the main loop."""
        _LOGGER.info("Stopping the main loop")
        self.running = False


class MainHandler(tornado.web.RequestHandler):
    """Tornado web request handler."""

    def initialize(self, data_queue):
        """Check if new predicted values are available in the queue before the get request."""
        try:
            model_list = data_queue.get_nowait()
            self.settings["model_list"] = model_list
        except EmptyQueueException:
            pass

    async def get(self):
        """Fetch and publish metric values asynchronously."""
        # update metric value on every request and publish the metric
        for predictor_model in self.settings["model_list"]:
            # get the current metric value so that it can be compared with the
            # predicted values
            current_metric_value = Metric(
                pc.get_current_metric_value(
                    metric_name=predictor_model.metric.metric_name,
                    label_config=predictor_model.metric.label_config,
                )[0]
            )

            metric_name = predictor_model.metric.metric_name
            prediction = predictor_model.predict_value(datetime.now())

            # Check for all the columns available in the prediction
            # and publish the values for each of them
            for column_name in list(prediction.columns):
                GAUGE_DICT[metric_name].labels(
                    **predictor_model.metric.label_config, value_type=column_name
                ).set(prediction[column_name][0])

            # Calculate for an anomaly (can be different for different models)
            anomaly = 1
            if (current_metric_value.metric_values["y"][0] < prediction["yhat_upper"][0]) \
                    and (current_metric_value.metric_values["y"][0] > prediction["yhat_lower"][0]):
                anomaly = 0

            # create a new time series that has value_type=anomaly
            # this value is 1 if an anomaly is found 0 if not
            GAUGE_DICT[metric_name].labels(
                **predictor_model.metric.label_config, value_type="anomaly"
            ).set(anomaly)

        self.write(generate_latest(REGISTRY).decode("utf-8"))
        self.set_header("Content-Type", "text; charset=utf-8")


class HealthHandler(tornado.web.RequestHandler):
    async def get(self):
        self.set_status(200)
        self.write("ok")


class LifecycleHandler(tornado.web.RequestHandler):
    def initialize(self, app):
        self.app = app

    async def post(self):
        self.set_status(200)
        self.write("ok")
        self.app.stop()
        ioloop = tornado.ioloop.IOLoop.instance()
        ioloop.add_callback(ioloop.stop)


async def main():
    parse_command_line()
    port = os.getenv("HTTP_PORT", "8080")

    # Queue to share data between the tornado server and the model training
    predicted_model_queue = Queue()
    parallelism = min(Configuration.parallelism, cpu_count())
    app = MainApplication(queue=predicted_model_queue)
    http_server = httpserver.HTTPServer(app)
    http_server.listen(int(port))

    # Initial run to generate metrics, before they are exposed
    app.train_model(initial_run=True, data_queue=predicted_model_queue)

    # Schedule the model training
    schedule.every(Configuration.retraining_interval_minutes).minutes.do(
        app.train_model, initial_run=False, data_queue=predicted_model_queue
    )
    _LOGGER.info(
        "Will retrain model every %s minutes", Configuration.retraining_interval_minutes
    )
    threading.Thread(target=app.loop_forever).start()
    await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        _LOGGER.error("Runtime error, exiting", e)
