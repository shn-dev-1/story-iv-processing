# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "iv_app" {
  name              = "/ecs/story-iv"
  retention_in_days = 30

  tags = {
    Name        = "${var.app_name}-log-group"
    Environment = var.environment
    Purpose     = "Image Generation Application Logs"
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "iv_app" {
  family                   = "${var.app_name}-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["EC2"]
  memory                   = var.memory

  execution_role_arn = data.terraform_remote_state.story_infra.outputs.ecs_task_execution_role_arn
  task_role_arn      = data.terraform_remote_state.story_infra.outputs.ecs_task_role_arn

  runtime_platform {
    cpu_architecture        = "X86_64"
    operating_system_family = "LINUX"
  }

  container_definitions = jsonencode([
    {
      name  = "${var.app_name}-container"
      image = "${aws_ecr_repository.iv_app.repository_url}:${var.app_image_tag}"

      portMappings = [
        {
          containerPort = var.app_port
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "QUEUE_URL"
          value = data.terraform_remote_state.story_infra.outputs.task_queue_urls["IMAGE"]
        },
        {
          name  = "AWS_REGION"
          value = var.aws_region
        },
        {
          name  = "MODEL_DIR"
          value = "/opt/models/wan2.2-ti2v-5b"
        },
        {
          name  = "DYNAMODB_TABLE"
          value = data.terraform_remote_state.story_infra.outputs.story_video_tasks_table_name
        }
      ]

      resourceRequirements = [
        {
          type  = "GPU"
          value = "1"
        }
      ]

      linuxParameters = {
        devices = [
          {
            hostPath      = "/dev/nvidia0"
            containerPath = "/dev/nvidia0"
            permissions   = ["read", "write"]
          }
        ]
      }

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.iv_app.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:${var.app_port}/healthz || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 120
      }

      essential = true
    }
  ])

  tags = {
    Name        = "${var.app_name}-task-definition"
    Environment = var.environment
    Purpose     = "Video Generation Application Task Definition"
  }
}

# EC2 Capacity Provider
resource "aws_ecs_capacity_provider" "iv_gpu" {
  name = "story-iv-gpu"

  auto_scaling_group_provider {
    auto_scaling_group_arn         = aws_autoscaling_group.iv_gpu.arn
    managed_termination_protection = "DISABLED"

    managed_scaling {
      maximum_scaling_step_size = 1
      minimum_scaling_step_size = 1
      status                    = "ENABLED"
      target_capacity           = 100
    }
  }

  tags = {
    Name        = "${var.app_name}-gpu-capacity-provider"
    Environment = var.environment
    Purpose     = "GPU Video Processing Capacity Provider"
  }
}

# Auto Scaling Group for GPU instances
resource "aws_autoscaling_group" "iv_gpu" {
  name                      = "${var.app_name}-gpu-asg"
  desired_capacity          = var.app_count
  max_size                  = var.app_count
  min_size                  = var.app_count
  target_group_arns         = []
  vpc_zone_identifier       = data.terraform_remote_state.story_infra.outputs.private_subnet_ids
  health_check_type         = "EC2"
  health_check_grace_period = 300

  launch_template {
    id      = aws_launch_template.iv_gpu.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "${var.app_name}-gpu-instance"
    propagate_at_launch = true
  }

  tag {
    key                 = "Environment"
    value               = var.environment
    propagate_at_launch = true
  }

  tag {
    key                 = "Purpose"
    value               = "GPU Video Processing Instance"
    propagate_at_launch = true
  }
}

# Launch Template for GPU instances
resource "aws_launch_template" "iv_gpu" {
  name_prefix   = "${var.app_name}-gpu-"
  image_id      = var.gpu_ami_id
  instance_type = var.gpu_instance_type

  network_interfaces {
    associate_public_ip_address = false
    security_groups             = [data.terraform_remote_state.story_infra.outputs.ecs_tasks_security_group_id]
    delete_on_termination       = true
  }

  iam_instance_profile {
    name = data.terraform_remote_state.story_infra.outputs.story_image_ec2_instance_profile_id
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    cluster_name = data.terraform_remote_state.story_infra.outputs.story_image_cluster_name
  }))

  monitoring {
    enabled = true
  }

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size = 75
      volume_type = "gp3"
    }
  }

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name        = "${var.app_name}-gpu-instance"
      Environment = var.environment
      Purpose     = "GPU Video Processing Instance"
    }
  }

  tag_specifications {
    resource_type = "volume"
    tags = {
      Name        = "${var.app_name}-gpu-volume"
      Environment = var.environment
      Purpose     = "GPU Video Processing Volume"
    }
  }
}

# ECS Service
resource "aws_ecs_service" "iv_app" {
  name            = "${var.app_name}-service"
  cluster         = data.terraform_remote_state.story_infra.outputs.story_image_cluster_name
  task_definition = aws_ecs_task_definition.iv_app.arn
  desired_count   = var.app_count

  network_configuration {
    security_groups  = [data.terraform_remote_state.story_infra.outputs.ecs_tasks_security_group_id]
    subnets          = data.terraform_remote_state.story_infra.outputs.private_subnet_ids
    assign_public_ip = false
  }

  capacity_provider_strategy {
    capacity_provider = aws_ecs_capacity_provider.iv_gpu.name
    weight            = 100
  }

  depends_on = [
    aws_ecs_task_definition.iv_app
  ]

  tags = {
    Name        = "${var.app_name}-ecs-service"
    Environment = var.environment
    Purpose     = "Image Generation Application ECS Service"
  }
}
