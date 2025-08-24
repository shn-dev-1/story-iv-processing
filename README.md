# Story IV Processing - Terraform Infrastructure

This repository contains the Terraform configuration for deploying the Story IV (Video Generation) processing service to AWS ECS.

## Overview

The Story IV Processing service is a containerized application that:
- Processes video generation jobs from SQS queues (with SNS wrapper support)
- Generates videos using Wan 2.2 TI2V-5B models
- Stores generated videos in S3
- Updates task statuses in DynamoDB
- Runs on AWS ECS EC2 with GPU instances (g5.xlarge - A10G 24GB VRAM)

## Architecture

- **ECS EC2**: Runs the containerized application on GPU instances
- **ECR**: Stores the Docker container images
- **SQS**: Receives video generation job requests (with SNS wrapper support)
- **S3**: Stores generated videos
- **DynamoDB**: Tracks task statuses and completion
- **CloudWatch**: Application logging and monitoring
- **VPC Endpoints**: For secure communication with AWS services
- **GPU Instances**: g5.xlarge with A10G 24GB VRAM for video processing
- **Auto Scaling Group**: Manages GPU instance lifecycle
- **NVIDIA Container Runtime**: Enables GPU access in containers

## Prerequisites

- AWS CLI configured with appropriate permissions
- Terraform >= 1.0
- Access to the shared infrastructure repository (`story-infra`)
- The following resources must exist in the shared infrastructure:
  - ECS Cluster
  - VPC with private subnets
  - SQS queues (including `VIDEO` queue for video generation)
  - S3 buckets for video storage
  - DynamoDB table for task tracking
  - IAM roles for ECS task execution and tasks
  - Security groups for ECS tasks
  - ECS instance profile for EC2 instances

## GitHub Secrets

For GitHub Actions workflows to function properly, the following secrets must be configured in your repository:

- **`AWS_ACCESS_KEY_ID`** - AWS access key for deployment
- **`AWS_SECRET_ACCESS_KEY`** - AWS secret key for deployment

These secrets are used by the GitHub Actions workflows to authenticate with AWS and deploy infrastructure and applications.

## Message Format

The application processes SQS messages that may be wrapped in SNS notifications. The expected message format is:

```json
{
  "parent_id": "12312312",
  "task_id": "12312311", 
  "prompt": "a cozy cabin in the forest, golden hour",
  "seed": 1234,             // optional
  "steps": 15,              // optional (default 15)
  "guidance": 7.0,          // optional (CFG)
  "width": 1280,            // optional (default 1280 for 720p)
  "height": 720,            // optional (default 720 for 720p)
  "num_videos": 1,          // optional, <= 2 recommended for GPU
  "negative_prompt": ""     // optional
}
```

### SNS Wrapper Support

When messages come through SNS, they are automatically unwrapped and validated. The application handles both direct SQS messages and SNS-wrapped messages seamlessly.

## Configuration

### Variables

Key configuration variables in `terraform.tfvars`:

- `aws_region`: AWS region (default: us-east-1)
- `environment`: Environment name (default: production)
- `app_name`: Application name (default: story-iv)
- `app_port`: Application port (default: 8080)
- `app_count`: Number of ECS service instances
- `gpu_ami_id`: AMI ID for GPU instances (Deep Learning AMI)
- `gpu_instance_type`: EC2 instance type (g5.xlarge - A10G 24GB VRAM)

**Note:** CPU and memory are not specified in the task definition since they are managed by the EC2 instance (g5.xlarge: 4 vCPU, 16GB RAM, 1 GPU).

**Note:** These values are automatically used by both Terraform infrastructure deployment and GitHub Actions application deployment, ensuring consistency across the entire deployment pipeline.

### Remote State

This configuration references the shared infrastructure via remote state:
- Backend: S3 bucket `story-service-terraform-state`
- Key: `story-iv-processing/terraform.tfstate`
- Infrastructure state: `terraform.tfstate` (shared resources)

## Deployment

### Automated Deployment via GitHub Actions

The repository includes GitHub Actions workflows for automated infrastructure and application deployment:

1. **`terraform.yml`** - Main deployment workflow
   - Runs automatically on merges to `main` branch
   - Deploys Terraform infrastructure
   - Builds and pushes Docker image to ECR
   - Deploys application to ECS
   - Requires `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` secrets

2. **`terraform-commit.yml`** - Feature branch validation
   - Runs on feature branches and pull requests
   - Performs Terraform plan only (no changes applied)
   - Comments results on PRs and commits

3. **`terraform-destroy.yml`** - Infrastructure cleanup
   - Manual trigger only
   - Requires typing "DESTROY" to confirm
   - Destroys all infrastructure resources

### Manual Deployment

For manual deployment or local development:

1. Initialize Terraform:
   ```bash
   terraform init
   ```

2. Review the plan:
   ```bash
   terraform plan
   ```

3. Apply the configuration:
   ```bash
   terraform apply
   ```

### Building and Pushing Docker Images

1. Build the ARM64 image:
   ```bash
   docker buildx build --platform linux/arm64 -t story-iv:latest .
   ```

2. Tag for ECR:
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $(terraform output -raw ecr_repository_url)
   docker tag story-iv:latest $(terraform output -raw ecr_repository_url):latest
   ```

3. Push to ECR:
   ```bash
   docker push $(terraform output -raw ecr_repository_url):latest
   ```

## Resources Created

- **ECR Repository**: `story-iv` for container images
- **ECS Task Definition**: Fargate task with ARM64 platform
- **ECS Service**: Service running the image generation tasks
- **CloudWatch Log Group**: `/ecs/story-iv` for application logs
- **Security Group**: Network access control for the application

## Terraform Outputs

The following values are exposed as Terraform outputs for use in CI/CD pipelines:

- **Configuration values**: `cpu`, `memory`, `app_port` - Used by GitHub Actions for task definition updates
- **Resource identifiers**: ECR repository, ECS service, security group ARNs and IDs
- **Infrastructure details**: Log group names, task definition family, etc.

## Monitoring

- **CloudWatch Logs**: Application logs at `/ecs/story-iv`
- **ECS Service**: Monitor task health and performance
- **SQS**: Monitor queue depth and processing times

## Security

- Tasks run in private subnets with no public IP
- VPC endpoints provide secure access to AWS services
- IAM roles follow least privilege principle
- Security groups restrict network access

## Troubleshooting

### Common Issues

1. **Task fails to start**: Check ECR repository access and image tags
2. **Container health check fails**: Verify the application is listening on port 8080
3. **SQS access denied**: Ensure task role has appropriate SQS permissions
4. **S3 access denied**: Verify task role has S3 read/write permissions

### Logs

Check CloudWatch logs for the application:
```bash
aws logs tail /ecs/story-iv --follow
```

## Contributing

1. Make changes to Terraform configuration
2. Run `terraform plan` to verify changes
3. Test in a non-production environment
4. Submit pull request with detailed description

## License

This project is part of the Story Service infrastructure.
