variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "app_name" {
  description = "Application name"
  type        = string
  default     = "story-iv"
}

variable "ecr_repository_name" {
  description = "Name of the ECR repository"
  type        = string
  default     = "story-iv"
}

variable "app_image_tag" {
  description = "Docker image tag to deploy"
  type        = string
  default     = "latest"
}

variable "app_port" {
  description = "Port the application listens on"
  type        = number
  default     = 8080
}

variable "app_count" {
  description = "Number of application instances to deploy"
  type        = number
  default     = 1
}



variable "gpu_ami_id" {
  description = "AMI ID for GPU instances (NVIDIA driver preinstalled)"
  type        = string
  default     = "ami-0c7217cdde317cfec" # Deep Learning AMI GPU PyTorch 2.1.0 (Amazon Linux 2) 20231213
}

variable "gpu_instance_type" {
  description = "EC2 instance type for GPU instances"
  type        = string
  default     = "g5.xlarge" # A10G 24GB VRAM
}
