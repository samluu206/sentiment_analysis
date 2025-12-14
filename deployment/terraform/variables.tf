variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "instance_type" {
  description = "EC2 instance type (t2.micro for free tier, t2.medium recommended)"
  type        = string
  default     = "t2.medium"

  validation {
    condition     = can(regex("^t[2-3]\\.(micro|small|medium|large)", var.instance_type))
    error_message = "Instance type must be a t2 or t3 instance (e.g., t2.micro, t2.medium, t3.medium)"
  }
}

variable "key_name" {
  description = "AWS EC2 key pair name for SSH access"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID where EC2 instance will be launched"
  type        = string
}

variable "subnet_id" {
  description = "Subnet ID where EC2 instance will be launched"
  type        = string
}

variable "allowed_ssh_cidrs" {
  description = "List of CIDR blocks allowed to SSH into the instance"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "root_volume_size" {
  description = "Size of root EBS volume in GB"
  type        = number
  default     = 20

  validation {
    condition     = var.root_volume_size >= 15 && var.root_volume_size <= 30
    error_message = "Root volume size must be between 15 GB and 30 GB (free tier limit is 30 GB)"
  }
}

variable "use_elastic_ip" {
  description = "Whether to allocate an Elastic IP for the instance"
  type        = bool
  default     = false
}
