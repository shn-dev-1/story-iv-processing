# ECR Repository outputs
output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.iv_app.repository_url
}

output "ecr_repository_name" {
  description = "Name of the ECR repository"
  value       = aws_ecr_repository.iv_app.name
}

output "ecr_repository_arn" {
  description = "ARN of the ECR repository"
  value       = aws_ecr_repository.iv_app.arn
}

# ECS outputs
output "ecs_service_name" {
  description = "Name of the ECS service"
  value       = aws_ecs_service.iv_app.name
}

output "ecs_service_id" {
  description = "ID of the ECS service"
  value       = aws_ecs_service.iv_app.id
}

output "ecs_task_definition_arn" {
  description = "ARN of the ECS task definition"
  value       = aws_ecs_task_definition.iv_app.arn
}

output "ecs_task_definition_family" {
  description = "Family of the ECS task definition"
  value       = aws_ecs_task_definition.iv_app.family
}

# CloudWatch outputs
output "log_group_name" {
  description = "Name of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.iv_app.name
}

output "log_group_arn" {
  description = "ARN of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.iv_app.arn
}

# Security Group outputs
output "security_group_id" {
  description = "ID of the security group"
  value       = aws_security_group.iv_app.id
}

output "security_group_arn" {
  description = "ARN of the security group"
  value       = aws_security_group.iv_app.arn
}



output "app_port" {
  description = "Port the application listens on"
  value       = var.app_port
}

output "memory" {
  description = "Memory for the task in MiB"
  value       = var.memory
}

# GPU Configuration outputs
output "gpu_ami_id" {
  description = "AMI ID for GPU instances"
  value       = var.gpu_ami_id
}

output "gpu_instance_type" {
  description = "EC2 instance type for GPU instances"
  value       = var.gpu_instance_type
}

output "capacity_provider_name" {
  description = "Name of the ECS capacity provider"
  value       = aws_ecs_capacity_provider.iv_gpu.name
}

output "auto_scaling_group_name" {
  description = "Name of the auto scaling group for GPU instances"
  value       = aws_autoscaling_group.iv_gpu.name
}

output "launch_template_name" {
  description = "Name of the launch template for GPU instances"
  value       = aws_launch_template.iv_gpu.name
}
