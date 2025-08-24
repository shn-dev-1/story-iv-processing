.PHONY: help init plan apply destroy build push logs clean

# Default target
help:
	@echo "Story IV Processing - Available Commands:"
	@echo ""
	@echo "Terraform Commands:"
	@echo "  init    - Initialize Terraform"
	@echo "  plan    - Show Terraform execution plan"
	@echo "  apply   - Apply Terraform configuration"
	@echo "  destroy - Destroy all Terraform resources"
	@echo "  output  - Show Terraform outputs"
	@echo ""
	@echo "Docker Commands:"
	@echo "  build   - Build x86_64 Docker image for GPU"
	@echo "  push    - Push image to ECR"
	@echo "  deploy  - Build and push image, then apply Terraform"
	@echo ""
	@echo "Utility Commands:"
	@echo "  logs    - Tail CloudWatch logs"
	@echo "  clean   - Clean up local files"
	@echo "  gpu-status - Show GPU instance status"
	@echo "  gpu-logs   - Tail GPU instance logs"

# Terraform commands
init:
	terraform init

plan: init
	terraform plan

apply: init
	terraform apply

destroy: init
	terraform destroy

output: init
	terraform output

# Docker commands
build:
	docker buildx build --platform linux/amd64 -t story-iv:latest .

push: build
	@echo "Logging into ECR..."
	@aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $$(terraform output -raw ecr_repository_url 2>/dev/null || echo "ECR repository not created yet. Run 'make apply' first.")
	@echo "Tagging image..."
	@docker tag story-iv:latest $$(terraform output -raw ecr_repository_url):latest
	@echo "Pushing to ECR..."
	@docker push $$(terraform output -raw ecr_repository_url):latest

deploy: push apply

# Utility commands
logs:
	@echo "Tailing CloudWatch logs for /ecs/story-iv..."
	@aws logs tail /ecs/story-iv --follow

clean:
	@echo "Cleaning up local files..."
	@rm -rf .terraform
	@rm -f .terraform.lock.hcl
	@rm -f *.tfstate*
	@rm -f *.tfplan*
	@echo "Cleanup complete"

# Development helpers
validate:
	terraform validate

fmt:
	terraform fmt -recursive

# Show current status
status:
	@echo "Current Terraform Status:"
	@terraform show -json 2>/dev/null | jq -r '.values.outputs | to_entries[] | "\(.key): \(.value.value)"' 2>/dev/null || echo "No resources deployed yet"

# GPU instance management
gpu-status:
	@echo "GPU Instance Status:"
	@aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names $$(terraform output -raw auto_scaling_group_name 2>/dev/null || echo "Auto scaling group not created yet") --query 'AutoScalingGroups[0].Instances[].[InstanceId,HealthStatus,LifecycleState]' --output table 2>/dev/null || echo "No GPU instances found"

gpu-logs:
	@echo "GPU Instance CloudWatch Logs:"
	@aws logs tail /ecs/story-iv --follow
