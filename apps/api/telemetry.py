from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
import os

def setup_telemetry():
    resource = Resource.create(attributes={
        "service.name": "rag-foundry",
    })
    
    # Connect to Phoenix OTel collector (default gRPC port 4317)
    # Inside docker: http://phoenix:4317
    # Local run: http://localhost:4317
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:4317")
    
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)
    
    # Use gRPC exporter
    otlp_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    
    span_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(span_processor)
    
    print(f"Telemetry initialized. Sending traces to {endpoint}")
