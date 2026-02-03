# Testing the Monitoring Stack

This guide shows you how to test the new Prometheus + Grafana monitoring implementation.

## Option 1: Local Testing (Recommended for Quick Validation)

### Step 1: Start the API

```bash
# Install dependencies if not already installed
pip install -e ".[api]"

# Start the FastAPI server
python -m uvicorn src.sentiment_analyzer.api.main:app --reload --port 8000
```

### Step 2: Check the Metrics Endpoint

Open a new terminal and run:

```bash
# Check that /metrics endpoint exists
curl http://localhost:8000/metrics
```

You should see Prometheus metrics in this format:
```
# HELP sentiment_predictions_total Total number of predictions made
# TYPE sentiment_predictions_total counter
sentiment_predictions_total{endpoint="predict",sentiment="POSITIVE"} 0.0
sentiment_predictions_total{endpoint="predict",sentiment="NEGATIVE"} 0.0

# HELP sentiment_prediction_duration_seconds Time spent processing prediction
# TYPE sentiment_prediction_duration_seconds histogram
...
```

### Step 3: Make Some Predictions

```bash
# Make a prediction to generate metrics
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!"}'

# Make a few more predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Terrible quality, waste of money"}'

# Make a batch prediction
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product!", "Not good", "Perfect!"]}'
```

### Step 4: Verify Metrics Are Being Collected

```bash
# Check metrics again
curl http://localhost:8000/metrics | grep sentiment_predictions_total
```

You should now see non-zero counts:
```
sentiment_predictions_total{endpoint="predict",sentiment="POSITIVE"} 2.0
sentiment_predictions_total{endpoint="predict",sentiment="NEGATIVE"} 1.0
sentiment_predictions_total{endpoint="batch_predict",sentiment="POSITIVE"} 2.0
sentiment_predictions_total{endpoint="batch_predict",sentiment="NEGATIVE"} 1.0
```

---

## Option 2: Docker Testing

### Step 1: Build Docker Images

```bash
# Build API image
docker build -t sentiment-api:test -f docker/api/Dockerfile .

# Test the image
docker run -d -p 8000:8000 --name test-api sentiment-api:test
```

### Step 2: Test Metrics

```bash
# Wait for container to start
sleep 5

# Check metrics endpoint
curl http://localhost:8000/metrics

# Make predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great product!"}'

# Verify metrics
curl http://localhost:8000/metrics | grep sentiment_predictions_total

# Cleanup
docker stop test-api && docker rm test-api
```

---

## Option 3: Full Kubernetes Testing

### Prerequisites

Ensure you have:
- K3s cluster running on EC2
- kubectl configured with cluster access
- Namespace created: `kubectl create namespace sentiment-analysis`

### Step 1: Deploy Monitoring Stack

```bash
# Deploy Prometheus
kubectl apply -f k8s/monitoring/prometheus-rbac.yaml
kubectl apply -f k8s/monitoring/prometheus-config.yaml
kubectl apply -f k8s/monitoring/prometheus-deployment.yaml
kubectl apply -f k8s/monitoring/prometheus-service.yaml

# Verify Prometheus is running
kubectl get pods -n sentiment-analysis | grep prometheus
kubectl logs -f deployment/prometheus -n sentiment-analysis
```

### Step 2: Deploy Grafana

```bash
# Deploy Grafana datasource config
kubectl apply -f k8s/monitoring/grafana-datasource.yaml

# Create dashboard ConfigMap from JSON file
kubectl create configmap grafana-dashboards \
  --from-file=sentiment-dashboard.json=k8s/monitoring/grafana-dashboard.json \
  --namespace=sentiment-analysis

# Deploy dashboard provider config
kubectl apply -f k8s/monitoring/grafana-dashboard-config.yaml

# Deploy Grafana
kubectl apply -f k8s/monitoring/grafana-deployment.yaml
kubectl apply -f k8s/monitoring/grafana-service.yaml

# Verify Grafana is running
kubectl get pods -n sentiment-analysis | grep grafana
kubectl logs -f deployment/grafana -n sentiment-analysis
```

### Step 3: Update API Deployment with Metrics Annotations

Add these annotations to your API deployment (`k8s/api-deployment.yaml`):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-api
  namespace: sentiment-analysis
spec:
  template:
    metadata:
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
```

Then apply:
```bash
kubectl apply -f k8s/api-deployment.yaml
```

### Step 4: Access Prometheus

```bash
# Get your EC2 public IP
EC2_IP=$(terraform -chdir=deployment/terraform output -raw ec2_public_ip)

# Or manually find it
echo "Access Prometheus at: http://<YOUR_EC2_IP>:30900"
```

Open browser and go to:
- Prometheus UI: `http://<EC2_IP>:30900`
- Targets: `http://<EC2_IP>:30900/targets`

**What to check:**
1. Go to Status → Targets
2. You should see `sentiment-api` target with status "UP"
3. If status is "DOWN", check:
   - API pod is running: `kubectl get pods -n sentiment-analysis`
   - Annotations are correct: `kubectl describe deployment sentiment-api -n sentiment-analysis`
   - Logs: `kubectl logs deployment/sentiment-api -n sentiment-analysis`

### Step 5: Query Metrics in Prometheus

In Prometheus UI, go to Graph tab and try these queries:

```promql
# Total predictions
sum(sentiment_predictions_total)

# Prediction rate (last 5 min)
sum(rate(sentiment_predictions_total[5m]))

# Average latency
rate(sentiment_prediction_duration_seconds_sum[5m]) / rate(sentiment_prediction_duration_seconds_count[5m])

# Positive vs negative ratio
sum(sentiment_predictions_total{sentiment="POSITIVE"}) / sum(sentiment_predictions_total)
```

### Step 6: Access Grafana

Open browser and go to: `http://<EC2_IP>:30300`

**Login:**
- Username: `admin`
- Password: `admin` (change on first login)

**What to check:**
1. Go to Configuration → Data Sources
2. You should see "Prometheus" datasource configured
3. Click "Test" to verify connection
4. Go to Dashboards
5. You should see "Sentiment Analysis Production Monitoring" dashboard
6. Open it to see all 12 panels

**If dashboard is not showing:**
```bash
# Check if ConfigMap was created
kubectl get configmap grafana-dashboards -n sentiment-analysis

# If not, create it manually
kubectl create configmap grafana-dashboards \
  --from-file=sentiment-dashboard.json=k8s/monitoring/grafana-dashboard.json \
  --namespace=sentiment-analysis

# Restart Grafana to pick up changes
kubectl rollout restart deployment/grafana -n sentiment-analysis
```

### Step 7: Generate Test Traffic

```bash
# Get API endpoint
API_URL="http://<EC2_IP>:30800"

# Make some predictions
for i in {1..10}; do
  curl -X POST $API_URL/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "Great product!"}'
done

for i in {1..5}; do
  curl -X POST $API_URL/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "Terrible experience"}'
done

# Make batch predictions
curl -X POST $API_URL/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Amazing!", "Awful", "Perfect", "Horrible", "Excellent"]}'
```

### Step 8: Verify Dashboard is Working

Go back to Grafana dashboard and check:

1. **Total Predictions (24h)**: Should show ~15+ predictions
2. **Prediction Rate**: Should show activity
3. **Model Status**: Should show "1" (model loaded)
4. **Predictions by Sentiment**: Should show lines for POSITIVE and NEGATIVE
5. **Sentiment Distribution**: Should show pie chart with ~67% positive, 33% negative
6. **API Latency**: Should show p50/p95/p99 latency lines
7. **Confidence Distribution**: Should show heatmap of confidence scores

---

## Troubleshooting

### Metrics endpoint returns 404

**Problem:** `/metrics` endpoint not found

**Solution:**
1. Check that prometheus-fastapi-instrumentator is installed
2. Verify main.py has this line: `Instrumentator().instrument(app).expose(app, endpoint="/metrics")`
3. Restart the API

### Prometheus shows target as DOWN

**Problem:** Prometheus can't scrape API metrics

**Solution:**
```bash
# Check API is running
kubectl get pods -n sentiment-analysis

# Check API logs
kubectl logs deployment/sentiment-api -n sentiment-analysis

# Check Prometheus config
kubectl get configmap prometheus-config -n sentiment-analysis -o yaml

# Verify API service exists
kubectl get svc sentiment-api -n sentiment-analysis

# Test metrics endpoint from inside cluster
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
  curl http://sentiment-api:8000/metrics
```

### Grafana dashboard is empty

**Problem:** Dashboard shows "No data"

**Solution:**
1. Check Prometheus datasource is configured:
   - Go to Configuration → Data Sources
   - Click "Prometheus"
   - Click "Test" (should say "Data source is working")
2. Verify metrics exist in Prometheus:
   - Go to Prometheus UI → Graph
   - Query: `sentiment_predictions_total`
   - Should return results
3. Generate some traffic to create metrics (see Step 7)
4. Refresh Grafana dashboard

### Dashboard not appearing in Grafana

**Problem:** "Sentiment Analysis Production Monitoring" dashboard not found

**Solution:**
```bash
# Check ConfigMaps exist
kubectl get configmap -n sentiment-analysis | grep grafana

# Recreate dashboard ConfigMap
kubectl create configmap grafana-dashboards \
  --from-file=sentiment-dashboard.json=k8s/monitoring/grafana-dashboard.json \
  --namespace=sentiment-analysis \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart Grafana
kubectl rollout restart deployment/grafana -n sentiment-analysis

# Wait for restart
kubectl rollout status deployment/grafana -n sentiment-analysis

# Check logs
kubectl logs deployment/grafana -n sentiment-analysis
```

---

## Expected Results

After successful testing, you should see:

1. **API Metrics Endpoint:**
   - `curl http://localhost:8000/metrics` returns Prometheus metrics
   - Metrics include: sentiment_predictions_total, sentiment_prediction_duration_seconds, etc.

2. **Prometheus:**
   - Target "sentiment-api" shows as UP
   - Queries return actual metric values
   - Graphs show time series data

3. **Grafana Dashboard:**
   - 12 panels with live data
   - Total predictions count increasing
   - Latency graphs showing p50/p95/p99
   - Sentiment distribution showing positive/negative ratio
   - No "No data" errors

---

## Next Steps

Once testing is successful:

1. Commit and push changes:
   ```bash
   git add .
   git commit -m "feat: add Prometheus and Grafana monitoring stack"
   git push origin test/ci-cd
   ```

2. Merge to main:
   ```bash
   git checkout main
   git merge test/ci-cd
   git push origin main
   ```

3. Deploy to production:
   - CI/CD will automatically build new Docker images
   - Manually deploy monitoring stack with kubectl
   - Verify dashboards are working in production

4. Set up alerts (future enhancement):
   - Add AlertManager configuration
   - Create alert rules for high latency, errors, etc.

---

## Quick Test Checklist

- [ ] API /metrics endpoint returns Prometheus metrics
- [ ] Predictions increment counters
- [ ] Latency histogram records values
- [ ] Confidence scores are tracked
- [ ] Batch size metrics work
- [ ] Model loaded gauge shows 1
- [ ] Errors are counted
- [ ] Prometheus scrapes API successfully
- [ ] Grafana datasource connects to Prometheus
- [ ] Dashboard loads with 12 panels
- [ ] Dashboard shows live data after generating traffic
