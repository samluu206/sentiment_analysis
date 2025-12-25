#!/bin/bash

set -e

echo "========================================="
echo "Building and Exporting Docker Image"
echo "========================================="

IMAGE_NAME="sentiment-analyzer"
IMAGE_TAG="latest"
EXPORT_FILE="sentiment-analyzer.tar"

echo "Step 1: Building Docker image"
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

echo "Step 2: Exporting image to tarball"
docker save ${IMAGE_NAME}:${IMAGE_TAG} -o ${EXPORT_FILE}

echo "Step 3: Compressing tarball"
gzip -f ${EXPORT_FILE}

echo "========================================="
echo "Image built and exported!"
echo "========================================="
echo ""
echo "Output file: ${EXPORT_FILE}.gz"
echo "Size: $(du -h ${EXPORT_FILE}.gz | cut -f1)"
echo ""
echo "To transfer to EC2:"
echo "scp -i your-key.pem ${EXPORT_FILE}.gz ubuntu@<EC2_IP>:~/"
echo ""
echo "On EC2, load the image:"
echo "gunzip ${EXPORT_FILE}.gz"
echo "sudo k3s ctr images import ${EXPORT_FILE}"