# Security & Compliance Checklist

## ✅ Data Protection

- [ ] Enable S3 bucket encryption (KMS)
- [ ] Enable SageMaker volume encryption
- [ ] Use VPC endpoints for private connectivity
- [ ] Enable CloudTrail logging for audit trail
- [ ] Implement data retention policies
- [ ] Use AWS Secrets Manager for credentials

## ✅ Network Security

- [ ] Deploy endpoints in private subnets
- [ ] Configure security groups (least privilege)
- [ ] Use VPC endpoints (no internet access)
- [ ] Enable VPC Flow Logs
- [ ] Implement network ACLs

## ✅ IAM & Access Control

- [ ] Use least privilege IAM policies
- [ ] Enable MFA for sensitive operations
- [ ] Rotate IAM credentials regularly
- [ ] Use IAM roles (not access keys)
- [ ] Enable CloudTrail for IAM audit

## ✅ Model Security

- [ ] Scan model files for vulnerabilities
- [ ] Validate model inputs (prevent injection)
- [ ] Implement rate limiting
- [ ] Monitor for model poisoning
- [ ] Version control for models

## ✅ Compliance (HIPAA, SOC2, etc.)

- [ ] Enable encryption at rest and in transit
- [ ] Implement access logs and audit trails
- [ ] Use dedicated instances (not shared)
- [ ] Implement data anonymization/pseudonymization
- [ ] Regular security assessments

## ✅ Monitoring & Incident Response

- [ ] Enable CloudWatch alarms
- [ ] Set up security event notifications
- [ ] Implement automated remediation
- [ ] Regular security patching
- [ ] Incident response plan
