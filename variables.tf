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

variable "cpu" {
  description = "CPU units for the task (2048 = 2 vCPU)"
  type        = number
  default     = 4096
}

variable "memory" {
  description = "Memory for the task in MiB (8192 = 8GB)"
  type        = number
  default     = 16384
}
