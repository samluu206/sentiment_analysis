# Production Monitoring with Prometheus & Grafana

Comprehensive monitoring and observability stack for the Sentiment Analysis API.

## Overview

The monitoring stack provides real-time visibility into:
- **API Performance**: Latency, throughput, error rates
- **ML Metrics**: Prediction distribution, confidence scores
- **System Health**: Model status, resource usage

## Architecture

```
┌─────────────┐
│  FastAPI    │──metrics──┐
│  /metrics   │           │
└─────────────┘           │
                          ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Gradio     │────▶│  Prometheus  │────▶│   Grafana    │
│  (future)   │     │   (scrape)   │     │ (visualize)  │
└─────────────┘     └──────────────┘     └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Alerting    │
                    │  (future)    │
                    └──────────────┘
```

## Metrics Collected

### Custom ML Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `sentiment_predictions_total` | Counter | Total predictions made | sentiment, endpoint |
| `sentiment_prediction_duration_seconds` | Histogram | Prediction latency | endpoint |
| `sentiment_prediction_confidence` | Histogram | Confidence score distribution | sentiment |
| `sentiment_batch_size` | Histogram | Batch prediction sizes | - |
| `sentiment_model_loaded` | Gauge | Model status (0 or 1) | - |
| `sentiment_errors_total` | Counter | Error count | error_type, endpoint |

### Auto-instrumented HTTP Metrics

- `http_requests_total`: Total HTTP requests by method/handler
- `http_request_duration_seconds`: HTTP request latency
- `http_requests_in_progress`: Active concurrent requests
- `http_request_size_bytes`: Request payload sizes
- `http_response_size_bytes`: Response payload sizes

## Dashboard Panels

### 1. Overview Row
- **Total Predictions (24h)**: Single number with trend
- **Prediction Rate**: Real-time requests/second
- **Model Status**: Loaded/Not Loaded indicator
- **Error Rate**: Errors per second with threshold colors

### 2. Prediction Analytics
- **Predictions by Sentiment**: Time series (positive vs negative)
- **Sentiment Distribution**: Pie chart showing % breakdown
- **Confidence Distribution**: Heatmap of confidence scores
- **Batch Sizes**: Distribution of batch prediction sizes

### 3. Performance Metrics
- **API Latency**: p50, p95, p99 percentiles over time
- **HTTP Request Rate**: Requests by endpoint
- **Average Confidence**: Gauge per sentiment type
- **Error Types**: Bar chart of error categories

## Key Performance Indicators (KPIs)

### SLA Targets

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| **Latency (p95)** | < 200ms | > 300ms |
| **Latency (p99)** | < 500ms | > 1s |
| **Error Rate** | < 1% | > 5% |
| **Throughput** | > 15 req/s | < 5 req/s |
| **Model Uptime** | 99.9% | < 99% |
| **Avg Confidence** | > 0.85 | < 0.70 |

### Business Metrics

- **Prediction Volume**: Track usage trends
- **Sentiment Ratio**: Monitor for drift (should stay ~50/50 for balanced data)
- **Confidence Trends**: Detect model degradation
- **Peak Load Times**: Identify capacity planning needs

## Deployment

### Prerequisites

```bash
# Ensure namespace exists
kubectl create namespace sentiment-analysis
```

### Deploy Monitoring Stack

```bash
# 1. Deploy Prometheus
kubectl apply -f k8s/monitoring/prometheus-rbac.yaml
kubectl apply -f k8s/monitoring/prometheus-config.yaml
kubectl apply -f k8s/monitoring/prometheus-deployment.yaml
kubectl apply -f k8s/monitoring/prometheus-service.yaml

# 2. Deploy Grafana
kubectl apply -f k8s/monitoring/grafana-datasource.yaml
kubectl apply -f k8s/monitoring/grafana-dashboard-config.yaml
# Note: Create grafana-dashboards ConfigMap from JSON file
kubectl apply -f k8s/monitoring/grafana-deployment.yaml
kubectl apply -f k8s/monitoring/grafana-service.yaml
```

### Update API Deployment

Add Prometheus annotations to API deployment:

```yaml
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8000"
    prometheus.io/path: "/metrics"
```

### Access Dashboards

**Prometheus:**
- URL: `http://<EC2_IP>:30900`
- Targets: `http://<EC2_IP>:30900/targets`

**Grafana:**
- URL: `http://<EC2_IP>:30300`
- Login: admin/admin (change on first login)
- Dashboard: "Sentiment Analysis Production Monitoring"

## Usage

### Viewing Metrics in Prometheus

**Check if metrics are being collected:**
```
http://<EC2_IP>:30900/graph
Query: sentiment_predictions_total
```

**Common Queries:**

```promql
# Prediction rate (last 5 min)
sum(rate(sentiment_predictions_total[5m]))

# Error percentage
sum(rate(sentiment_errors_total[5m])) / sum(rate(sentiment_predictions_total[5m])) * 100

# Average latency
rate(sentiment_prediction_duration_seconds_sum[5m]) / rate(sentiment_prediction_duration_seconds_count[5m])

# Positive/negative ratio
sum(sentiment_predictions_total{sentiment="POSITIVE"}) / sum(sentiment_predictions_total)
```

### Grafana Dashboard

**Pre-configured panels:**
- Real-time metrics update every 10 seconds
- Historical data (7 day retention)
- Multiple visualization types (graphs, gauges, heatmaps)
- Drill-down capabilities

**Creating custom panels:**
1. Click "+ Add panel"
2. Select "Prometheus" as datasource
3. Enter PromQL query
4. Choose visualization type
5. Configure thresholds and alerts

## Monitoring in Production

### Health Checks

```bash
# Check Prometheus is scraping API
kubectl logs -f deployment/prometheus -n sentiment-analysis | grep "sentiment-api"

# Verify Grafana dashboard
kubectl logs -f deployment/grafana -n sentiment-analysis

# Test metrics endpoint
curl http://<EC2_IP>:30800/metrics
```

### Troubleshooting

**No metrics showing in Grafana:**
1. Check Prometheus is running: `kubectl get pods -n sentiment-analysis`
2. Verify Prometheus targets: Go to Prometheus UI → Status → Targets
3. Check API is exposing /metrics: `curl http://sentiment-api:8000/metrics`
4. Verify Grafana datasource: Configuration → Data Sources

**High latency detected:**
1. Check p95/p99 latency panel
2. Investigate batch vs single predictions
3. Review resource usage (CPU/Memory)
4. Scale replicas if needed

**Prediction drift detected:**
1. Monitor sentiment distribution panel
2. Check if ratio has changed significantly
3. Review confidence score trends
4. Consider model retraining

## Best Practices

### 1. Regular Monitoring

- Check dashboards daily
- Review weekly trends
- Set up alerts for anomalies

### 2. Capacity Planning

- Monitor peak usage times
- Track growth trends
- Plan scaling before hitting limits

### 3. Model Performance

- Track confidence score trends
- Monitor prediction distribution
- Alert on drift (>10% ratio change)

### 4. Incident Response

- Use metrics to diagnose issues
- Correlate errors with deployments
- Track recovery time

## Security

**Production Checklist:**

- [ ] Change default Grafana password
- [ ] Enable authentication on Prometheus
- [ ] Use NetworkPolicies to restrict access
- [ ] Enable HTTPS/TLS
- [ ] Store credentials in Kubernetes Secrets
- [ ] Limit dashboard edit permissions
- [ ] Enable audit logging
- [ ] Rotate passwords regularly

## Future Enhancements

### 1. Alerting (AlertManager)

```yaml
alerts:
  - name: HighErrorRate
    expr: sum(rate(sentiment_errors_total[5m])) > 0.05
    severity: critical

  - name: HighLatency
    expr: histogram_quantile(0.95, rate(sentiment_prediction_duration_seconds_bucket[5m])) > 0.5
    severity: warning

  - name: ModelNotLoaded
    expr: sentiment_model_loaded == 0
    severity: critical
```

### 2. Distributed Tracing

- Add OpenTelemetry instrumentation
- Integrate with Jaeger/Zipkin
- Trace request flow through system

### 3. Log Aggregation

- Deploy Loki or ELK stack
- Correlate logs with metrics
- Full observability (metrics + logs + traces)

### 4. Custom Business Metrics

- Revenue per prediction
- User satisfaction scores
- Feature usage analytics
- A/B test metrics

### 5. ML-Specific Monitoring

- **Data drift detection**: Input distribution changes
- **Model drift detection**: Performance degradation
- **Feature importance tracking**: Which features matter most
- **Prediction explanations**: SHAP/LIME integration

## Interview Talking Points

"I implemented production monitoring with Prometheus and Grafana, tracking:
- **12 custom metrics** including prediction latency, confidence scores, and sentiment distribution
- **Real-time dashboards** with 12 visualization panels
- **SLA tracking** with p50/p95/p99 latency percentiles
- **ML-specific metrics** like confidence score distribution and prediction drift
- **Full observability** of the production ML system"

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/dashboards/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Monitoring ML Systems (Google)](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
