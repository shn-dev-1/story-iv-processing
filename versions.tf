terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "story-service-terraform-state"
    key            = "story-iv-processing/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "story-terraform-lock"
    encrypt        = true
  }
}

# Remote state data source for infrastructure values
data "terraform_remote_state" "story_infra" {
  backend = "s3"
  config = {
    bucket         = "story-service-terraform-state"
    key            = "terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "story-terraform-lock"
  }
}
