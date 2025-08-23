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

# Configuration outputs for GitHub Actions
output "cpu" {
  description = "CPU units for the task"
  value       = var.cpu
}

output "memory" {
  description = "Memory for the task in MiB"
  value       = var.memory
}

output "app_port" {
  description = "Port the application listens on"
  value       = var.app_port
}
