# Security Group for Image Generation Application
resource "aws_security_group" "iv_app" {
  name        = "${var.app_name}-security-group"
  description = "Security group for Image Generation application ECS tasks"
  vpc_id      = data.terraform_remote_state.story_infra.outputs.vpc_id

  # Allow outbound traffic to VPC endpoints and S3
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }

  # Allow health check from ECS tasks
  ingress {
    from_port       = var.app_port
    to_port         = var.app_port
    protocol        = "tcp"
    security_groups = [data.aws_security_group.ecs_tasks.id]
    description     = "Allow health checks from ECS tasks"
  }

  tags = {
    Name        = "${var.app_name}-security-group"
    Environment = var.environment
    Purpose     = "Image Generation Application Security Group"
  }
}

# Security Group for ECS Tasks (referenced from existing infrastructure)
# This is a data source to reference the existing security group
data "aws_security_group" "ecs_tasks" {
  name   = "ecs-tasks-sg"
  vpc_id = data.terraform_remote_state.story_infra.outputs.vpc_id
}
