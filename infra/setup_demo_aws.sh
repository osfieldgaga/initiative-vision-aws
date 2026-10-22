#!/usr/bin/env bash
# AWS resources for the public demo (initiativelabs.cc/ivision-demo -> demo.initiativelabs.cc).
# Run once, from a machine with admin AWS credentials:
#
#   INSTANCE_ID=i-0123... bash infra/setup_demo_aws.sh
#
# VERCEL_TEAM_SLUG: Vercel dashboard -> the team's Settings -> General -> "Team URL" (vercel.com/<slug>).
# Creates (and prints the values to put in Vercel's environment variables):
#   1. Vercel as an OIDC identity provider, so the website gets AWS credentials without stored keys
#   2. role ivision-web: assumed by the website (project initiative-labs, production only); may
#      start/describe the demo instance and assume role 3
#   3. role ivision-judge-uploader: what a judge's virtual Pi gets for 1 h; may only put frames
#      under live/ and send to the frame queue (the real Pi's permissions)
#   4. an Elastic IP on the instance, for the DNS record demo.initiativelabs.cc
set -euo pipefail

: "${INSTANCE_ID:?set INSTANCE_ID}"
VERCEL_TEAM_SLUG=${VERCEL_TEAM_SLUG:-osfieldgaga}   # from the site's Vercel OIDC token (iss)
PROJECT=${VERCEL_PROJECT:-initiative-labs}
REGION=us-east-1
BUCKET=initiative-vision-1
QUEUE=lld-frames.fifo
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
ISSUER_HOST="oidc.vercel.com/${VERCEL_TEAM_SLUG}"
PROVIDER_ARN="arn:aws:iam::${ACCOUNT}:oidc-provider/${ISSUER_HOST}"
QUEUE_ARN="arn:aws:sqs:${REGION}:${ACCOUNT}:${QUEUE}"
QUEUE_URL="https://sqs.${REGION}.amazonaws.com/${ACCOUNT}/${QUEUE}"

echo "== 1. Vercel OIDC provider ($ISSUER_HOST)"
if ! aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$PROVIDER_ARN" >/dev/null 2>&1; then
  aws iam create-open-id-connect-provider --url "https://${ISSUER_HOST}" \
    --client-id-list "https://vercel.com/${VERCEL_TEAM_SLUG}" >/dev/null
fi

echo "== 2. role ivision-web"
cat > /tmp/ivision-web-trust.json <<EOF
{"Version": "2012-10-17", "Statement": [{
  "Effect": "Allow",
  "Principal": {"Federated": "${PROVIDER_ARN}"},
  "Action": "sts:AssumeRoleWithWebIdentity",
  "Condition": {"StringEquals": {
    "${ISSUER_HOST}:aud": "https://vercel.com/${VERCEL_TEAM_SLUG}",
    "${ISSUER_HOST}:sub": "owner:${VERCEL_TEAM_SLUG}:project:${PROJECT}:environment:production"}}}]}
EOF
aws iam get-role --role-name ivision-web >/dev/null 2>&1 \
  || aws iam create-role --role-name ivision-web --assume-role-policy-document file:///tmp/ivision-web-trust.json >/dev/null
aws iam update-assume-role-policy --role-name ivision-web --policy-document file:///tmp/ivision-web-trust.json
aws iam put-role-policy --role-name ivision-web --policy-name ivision-web --policy-document "{
  \"Version\": \"2012-10-17\", \"Statement\": [
    {\"Effect\": \"Allow\", \"Action\": \"ec2:StartInstances\",
     \"Resource\": \"arn:aws:ec2:${REGION}:${ACCOUNT}:instance/${INSTANCE_ID}\"},
    {\"Effect\": \"Allow\", \"Action\": \"ec2:DescribeInstances\", \"Resource\": \"*\"},
    {\"Effect\": \"Allow\", \"Action\": \"sts:AssumeRole\",
     \"Resource\": \"arn:aws:iam::${ACCOUNT}:role/ivision-judge-uploader\"}]}"

echo "== 3. role ivision-judge-uploader"
cat > /tmp/ivision-judge-trust.json <<EOF
{"Version": "2012-10-17", "Statement": [{
  "Effect": "Allow",
  "Principal": {"AWS": "arn:aws:iam::${ACCOUNT}:role/ivision-web"},
  "Action": "sts:AssumeRole"}]}
EOF
aws iam get-role --role-name ivision-judge-uploader >/dev/null 2>&1 \
  || aws iam create-role --role-name ivision-judge-uploader --assume-role-policy-document file:///tmp/ivision-judge-trust.json >/dev/null
aws iam update-assume-role-policy --role-name ivision-judge-uploader --policy-document file:///tmp/ivision-judge-trust.json
aws iam put-role-policy --role-name ivision-judge-uploader --policy-name ivision-judge-uploader --policy-document "{
  \"Version\": \"2012-10-17\", \"Statement\": [
    {\"Effect\": \"Allow\", \"Action\": \"s3:PutObject\", \"Resource\": \"arn:aws:s3:::${BUCKET}/live/*\"},
    {\"Effect\": \"Allow\", \"Action\": \"sqs:SendMessage\", \"Resource\": \"${QUEUE_ARN}\"}]}"

echo "== 4. Elastic IP"
EIP=$(aws ec2 describe-addresses --region "$REGION" --filters "Name=instance-id,Values=${INSTANCE_ID}" \
      --query 'Addresses[0].PublicIp' --output text)
if [ "$EIP" = "None" ] || [ -z "$EIP" ]; then
  ALLOC=$(aws ec2 allocate-address --region "$REGION" --domain vpc --query AllocationId --output text)
  aws ec2 associate-address --region "$REGION" --instance-id "$INSTANCE_ID" --allocation-id "$ALLOC" >/dev/null
  EIP=$(aws ec2 describe-addresses --region "$REGION" --allocation-ids "$ALLOC" --query 'Addresses[0].PublicIp' --output text)
fi

cat <<EOF

Done. Next:
  Porkbun DNS:  A record  demo.initiativelabs.cc -> ${EIP}
  Vercel env vars (project ${PROJECT}, Production):
    AWS_ROLE_ARN=arn:aws:iam::${ACCOUNT}:role/ivision-web
    IVISION_JUDGE_ROLE_ARN=arn:aws:iam::${ACCOUNT}:role/ivision-judge-uploader
    IVISION_INSTANCE_ID=${INSTANCE_ID}
    IVISION_QUEUE_URL=${QUEUE_URL}
    IVISION_DEMO_PASSWORD=<choose one>
    LIVE_TOKEN_SECRET=<same value as LIVE_TOKEN_SECRET in /etc/lld.env on the instance>
EOF
