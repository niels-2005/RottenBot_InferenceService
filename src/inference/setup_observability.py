import logging

from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from src.config import Config


def setup_observability(service_name: str = "my-app"):
    """Sets up OpenTelemetry observability components for tracing, logging, and metrics.

    Initializes and configures OpenTelemetry providers for distributed tracing,
    structured logging, and metrics collection. Exports data to an OTLP endpoint
    (e.g., Jaeger or Prometheus) running on localhost:4317. Also sets up a Python
    logging handler to integrate with OpenTelemetry logs.

    Args:
        service_name (str): The name of the service for resource identification.
            Defaults to "my-app".

    Returns:
        tuple: A tuple containing the tracer and meter instances for the current module.
            - tracer: An OpenTelemetry tracer for creating spans.
            - meter: An OpenTelemetry meter for creating metrics.
    """
    resource = Resource.create({"service.name": service_name})

    # Tracing Setup
    trace_provider = TracerProvider(resource=resource)
    trace_exporter = OTLPSpanExporter(
        endpoint=Config.OBSERVABILITY_ENDPOINT, insecure=True
    )
    trace_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
    trace.set_tracer_provider(trace_provider)

    # Logging Setup
    logger_provider = LoggerProvider(resource=resource)
    log_exporter = OTLPLogExporter(
        endpoint=Config.OBSERVABILITY_ENDPOINT, insecure=True
    )
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    set_logger_provider(logger_provider)

    # Python logging Handler
    handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

    # Metrics Setup
    metric_exporter = OTLPMetricExporter(
        endpoint=Config.OBSERVABILITY_ENDPOINT, insecure=True
    )
    metric_reader = PeriodicExportingMetricReader(
        metric_exporter, export_interval_millis=15000
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)


# Helper function for other modules
def get_tracer(module_name: str):
    return trace.get_tracer(module_name)


def get_meter(module_name: str):
    return metrics.get_meter(module_name)
