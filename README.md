# Story IV Processing - Terraform Infrastructure

This repository contains the Terraform configuration for deploying the Story IV (Image Generation) processing service to AWS ECS.

## Overview

The Story IV Processing service is a containerized application that:
- Processes image generation jobs from SQS queues
- Generates images using Stable Diffusion ONNX models
- Stores generated images in S3
- Runs on AWS ECS Fargate with ARM64 (Graviton) processors

## Architecture

- **ECS Fargate**: Runs the containerized application
- **ECR**: Stores the Docker container images
- **SQS**: Receives image generation job requests
- **S3**: Stores generated images
- **CloudWatch**: Application logging and monitoring
- **VPC Endpoints**: For secure communication with AWS services

## Prerequisites

- AWS CLI configured with appropriate permissions
- Terraform >= 1.0
- Access to the shared infrastructure repository (`story-infra`)
- The following resources must exist in the shared infrastructure:
  - ECS Cluster
  - VPC with private subnets
  - SQS queues (including `IV` queue for image generation)
  - S3 buckets for image storage
  - IAM roles for ECS task execution and tasks
  - Security groups for ECS tasks

## GitHub Secrets

For GitHub Actions workflows to function properly, the following secrets must be configured in your repository:

- **`AWS_ACCESS_KEY_ID`** - AWS access key for deployment
- **`AWS_SECRET_ACCESS_KEY`** - AWS secret key for deployment

These secrets are used by the GitHub Actions workflows to authenticate with AWS and deploy infrastructure and applications.

## Configuration

### Variables

Key configuration variables in `terraform.tfvars`:

- `aws_region`: AWS region (default: us-east-1)
- `environment`: Environment name (default: production)
- `app_name`: Application name (default: story-iv)
- `cpu`: CPU units (2048 = 2 vCPU for ARM64)
- `memory`: Memory in MiB (8192 = 8GB)
- `app_count`: Number of ECS service instances

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
