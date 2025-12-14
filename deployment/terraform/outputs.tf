output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.k3s_node.id
}

output "instance_public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = var.use_elastic_ip ? aws_eip.k3s_eip[0].public_ip : aws_instance.k3s_node.public_ip
}

output "instance_private_ip" {
  description = "Private IP address of the EC2 instance"
  value       = aws_instance.k3s_node.private_ip
}

output "security_group_id" {
  description = "Security group ID"
  value       = aws_security_group.k3s_sg.id
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ${var.key_name}.pem ubuntu@${var.use_elastic_ip ? aws_eip.k3s_eip[0].public_ip : aws_instance.k3s_node.public_ip}"
}

output "api_url" {
  description = "FastAPI endpoint URL"
  value       = "http://${var.use_elastic_ip ? aws_eip.k3s_eip[0].public_ip : aws_instance.k3s_node.public_ip}:30800"
}

output "gradio_url" {
  description = "Gradio web interface URL"
  value       = "http://${var.use_elastic_ip ? aws_eip.k3s_eip[0].public_ip : aws_instance.k3s_node.public_ip}:30786"
}

output "swagger_ui_url" {
  description = "Swagger UI URL for API documentation"
  value       = "http://${var.use_elastic_ip ? aws_eip.k3s_eip[0].public_ip : aws_instance.k3s_node.public_ip}:30800/docs"
}
