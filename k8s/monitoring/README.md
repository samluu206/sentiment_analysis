# Monitoring Stack Deployment

This directory contains Prometheus and Grafana configurations for monitoring the Sentiment Analysis API.

## Components

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards

## Quick Start

### 1. Deploy Monitoring Stack

```bash
# Apply all monitoring resources
kubectl apply -f k8s/monitoring/
```

### 2. Access Dashboards

**Prometheus:**
- URL: `http://<EC2_IP>:30900`
- No authentication required

**Grafana:**
- URL: `http://<EC2_IP>:30300`
- Username: `admin`
- Password: `admin` (change after first login)

### 3. Import Dashboard (if not auto-loaded)

If the dashboard doesn't appear automatically:
1. Go to Grafana → Dashboards → Import
2. Upload `grafana-dashboard.json`
3. Select "Prometheus" as the data source

## Metrics Collected

### API Metrics
- `sentiment_predictions_total`: Total predictions by sentiment and endpoint
- `sentiment_prediction_duration_seconds`: Prediction latency histogram
- `sentiment_prediction_confidence`: Confidence score distribution
- `sentiment_batch_size`: Batch size distribution
- `sentiment_model_loaded`: Model status (0 or 1)
- `sentiment_errors_total`: Error count by type

### HTTP Metrics (auto-instrumented)
- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request latency
- `http_requests_in_progress`: Active requests

## Dashboard Panels

1. **Total Predictions**: 24h prediction count
2. **Prediction Rate**: Real-time req/s
3. **Model Status**: Whether model is loaded
4. **Error Rate**: Errors per second
5. **Predictions by Sentiment**: Time series of positive/negative predictions
6. **Sentiment Distribution**: Pie chart of prediction distribution
7. **API Latency**: p50, p95, p99 percentiles
8. **Confidence Score Distribution**: Heatmap of confidence scores
9. **Batch Size Distribution**: Batch prediction sizes
10. **Error Types**: Bar chart of error types
11. **HTTP Request Rate**: Request rate by endpoint
12. **Average Confidence**: Gauge showing avg confidence per sentiment

## Troubleshooting

### Prometheus Not Scraping API

Check if API has metrics endpoint:
```bash
curl http://<API_IP>:8000/metrics
```

Check Prometheus targets:
```bash
# Access Prometheus UI
# Go to Status → Targets
```

### Grafana Dashboard Empty

1. Verify Prometheus datasource is configured
2. Check that Prometheus is collecting metrics
3. Query Prometheus directly: `sum(sentiment_predictions_total)`

### No Metrics Showing

Ensure API annotations are set:
```yaml
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8000"
    prometheus.io/path: "/metrics"
```

## Deployment Order

1. Namespace (if not exists)
2. Prometheus RBAC
3. Prometheus ConfigMap
4. Prometheus Deployment + Service
5. Grafana ConfigMaps (datasource + dashboards)
6. Grafana Deployment + Service

## Storage

Currently using `emptyDir` for both Prometheus and Grafana (data lost on pod restart).

**For production**, use PersistentVolumes:

```yaml
volumes:
- name: prometheus-storage
  persistentVolumeClaim:
    claimName: prometheus-pvc
```

## Resource Usage

**Prometheus:**
- Requests: 250m CPU, 512Mi RAM
- Limits: 500m CPU, 1Gi RAM
- Storage: ~100MB/day (7 day retention)

**Grafana:**
- Requests: 100m CPU, 256Mi RAM
- Limits: 200m CPU, 512Mi RAM

## Security Considerations

**IMPORTANT for Production:**

1. **Change default Grafana password**
2. **Enable authentication on Prometheus** (use reverse proxy with auth)
3. **Use HTTPS** with Let's Encrypt
4. **Restrict access** with NetworkPolicies
5. **Use Secrets** instead of ConfigMaps for passwords

Example:
```bash
kubectl create secret generic grafana-admin \
  --from-literal=admin-user=admin \
  --from-literal=admin-password=<strong-password> \
  -n sentiment-analysis
```

## Monitoring Best Practices

### Alerting (Future Enhancement)

Add Prometheus AlertManager for:
- High error rate (> 5%)
- High latency (p95 > 500ms)
- Model not loaded
- Low prediction rate (potential downtime)

### Retention

- **Development**: 7 days
- **Production**: 30 days minimum

### Backups

Export important dashboards:
```bash
# From Grafana UI: Dashboard → Share → Export → Save to file
```

## Custom Queries

### Prediction Rate (last 5 min)
```promql
sum(rate(sentiment_predictions_total[5m]))
```

### Error Percentage
```promql
sum(rate(sentiment_errors_total[5m])) / sum(rate(sentiment_predictions_total[5m])) * 100
```

### Average Latency
```promql
rate(sentiment_prediction_duration_seconds_sum[5m]) / rate(sentiment_prediction_duration_seconds_count[5m])
```

### Positive vs Negative Ratio
```promql
sum(sentiment_predictions_total{sentiment="POSITIVE"}) / sum(sentiment_predictions_total)
```

## Next Steps

1. **Set up AlertManager** for proactive monitoring
2. **Add custom alerts** for critical metrics
3. **Configure retention policies** based on needs
4. **Integrate with logging** (ELK/Loki) for full observability
5. **Add distributed tracing** with Jaeger/Zipkin
