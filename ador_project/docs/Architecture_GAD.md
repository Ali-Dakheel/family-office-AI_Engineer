# Global Architecture Document (GAD)
## ADOR - Augmented Document Reader System

**Document Type:** Global Architecture Document (GAD)  
**Author:** Ali Dakheel 
**Organization:** CMI Architecture & Innovation Team  
**Version:** 1.0  
**Classification:** Internal Use

---

## 1. Executive Summary

This Global Architecture Document describes the design and integration of the Augmented Document Reader (ADOR) system into the CMI Information System (IS) ecosystem. ADOR is an AI-powered document processing platform that extracts, classifies, and analyzes financial entities from various document formats including DOCX, PDF, and chat transcripts.

**Strategic Value:**
- **Efficiency:** Reduces manual document review time by 80%
- **Accuracy:** Achieves >95% entity extraction accuracy
- **Compliance:** Provides audit trails for regulatory requirements
- **Scalability:** Processes 1000+ documents per day

**Key Capabilities:**
- Multi-format document ingestion (DOCX, PDF, TXT, chat logs)
- Intelligent routing to appropriate extraction methods (rule-based, NER, LLM)
- RESTful API for programmatic access
- Web UI for manual document uploads
- Real-time and batch processing modes

---

## 2. System Context

### 2.1 High-Level Context Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CMI Information System                       │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐            │
│  │  Trading     │  │   Risk Mgmt   │  │  Compliance  │            │
│  │  Systems     │  │   Systems     │  │  Systems     │            │
│  └──────┬───────┘  └───────┬───────┘  └──────┬───────┘            │
│         │                   │                  │                    │
│         └───────────────────┼──────────────────┘                    │
│                             │                                       │
│                    ┌────────v───────────┐                          │
│                    │   API Gateway      │                          │
│                    │  (Authentication)  │                          │
│                    └────────┬───────────┘                          │
│                             │                                       │
└─────────────────────────────┼───────────────────────────────────────┘
                              │
                    ┌─────────v──────────┐
                    │                    │
              ┌─────┴─────┐        ┌────┴────┐
              │ API Access │        │   UI    │
              │ (Headless) │        │ (Human) │
              └─────┬─────┘         └────┬────┘
                    │                    │
                    └─────────┬──────────┘
                              │
              ┌───────────────v────────────────┐
              │      ADOR System               │
              │  ┌─────────────────────────┐   │
              │  │  Document Processing    │   │
              │  │  Engine                 │   │
              │  └────────┬────────────────┘   │
              │           │                    │
              │  ┌────────v────────────────┐   │
              │  │  Rule | NER | LLM       │   │
              │  │  Extractors             │   │
              │  └─────────────────────────┘   │
              └────────────────────────────────┘
                              │
                    ┌─────────v──────────┐
                    │  Document Store    │
                    │  Result Database   │
                    └────────────────────┘
```

### 2.2 Stakeholders

| Stakeholder | Interest | Requirements |
|-------------|----------|--------------|
| **Trading Desk** | Fast term sheet analysis | <60s processing, real-time API |
| **Risk Management** | Accurate entity extraction | >95% accuracy, audit trails |
| **Compliance** | Regulatory reporting | Complete extraction logs |
| **IT Operations** | System reliability | 99.5% uptime, monitoring |
| **Data Security** | Confidentiality | Encryption, access control |

---

## 3. Functional Architecture

### 3.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            ADOR Architecture                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Presentation Layer                         │   │
│  │  ┌──────────────┐        ┌───────────────┐                     │   │
│  │  │   Web UI     │        │   REST API    │                     │   │
│  │  │  (React)     │        │  (FastAPI)    │                     │   │
│  │  └──────┬───────┘        └───────┬───────┘                     │   │
│  └─────────┼────────────────────────┼─────────────────────────────┘   │
│            │                        │                                  │
│  ┌─────────v────────────────────────v─────────────────────────────┐   │
│  │                      Application Layer                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │   │
│  │  │  Document    │  │  Workflow    │  │   Feature    │         │   │
│  │  │  Classifier  │  │  Orchestrator│  │   Router     │         │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │   │
│  └─────────┼────────────────┼─────────────────┼──────────────────┘   │
│            │                │                  │                       │
│  ┌─────────v────────────────v──────────────────v──────────────────┐   │
│  │                   Processing Layer                              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │   │
│  │  │  Rule-Based  │  │  NER Model   │  │  LLM + RAG   │         │   │
│  │  │  Parser      │  │  Extractor   │  │  Pipeline    │         │   │
│  │  │  (DOCX)      │  │  (Chat/TXT)  │  │  (PDF)       │         │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │   │
│  └─────────┼────────────────┼─────────────────┼──────────────────┘   │
│            │                │                  │                       │
│  ┌─────────v────────────────v──────────────────v──────────────────┐   │
│  │                      Data Layer                                 │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │   │
│  │  │  Document    │  │  Vector DB   │  │  Results DB  │         │   │
│  │  │  Storage     │  │  (ChromaDB)  │  │  (PostgreSQL)│         │   │
│  │  │  (S3/MinIO)  │  └──────────────┘  └──────────────┘         │   │
│  │  └──────────────┘                                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   Infrastructure Layer                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │   │
│  │  │  Message     │  │  Monitoring  │  │  Security    │         │   │
│  │  │  Queue       │  │  (Prometheus)│  │  (Vault)     │         │   │
│  │  │  (RabbitMQ)  │  └──────────────┘  └──────────────┘         │   │
│  │  └──────────────┘                                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Descriptions

#### **3.2.1 Presentation Layer**

**Web UI (React + TypeScript)**
- User-friendly interface for document upload
- Real-time extraction progress tracking
- Results visualization and export
- User authentication and authorization

**REST API (FastAPI)**
- OpenAPI specification for programmatic access
- JWT-based authentication
- Rate limiting and throttling
- WebSocket support for real-time updates

#### **3.2.2 Application Layer**

**Document Classifier**
- Determines document type from extension and content analysis
- Routes to appropriate extraction pipeline
- Handles multi-format documents (e.g., PDF with embedded tables)

**Workflow Orchestrator**
- Manages extraction lifecycle (queued → processing → complete)
- Handles retries and error recovery
- Coordinates multi-stage processing (e.g., LLM multi-pass extraction)

**Feature Router**
- Directs requests to classification, summarization, NER, or Q&A features
- Currently focused on NER for MVP
- Extensible for future capabilities

#### **3.2.3 Processing Layer**

**Rule-Based Parser (DOCX)**
- Uses python-docx for document structure parsing
- Regex-based pattern matching for entities
- Deterministic, high-speed extraction
- Best for: Structured templates, term sheets, contracts

**NER Model Extractor (Chat/TXT)**
- Hybrid approach: spaCy NER + custom financial patterns
- Fine-tuned on financial entity types
- Confidence scoring for each extraction
- Best for: Semi-structured chat logs, messages, emails

**LLM + RAG Pipeline (PDF)**
- Retrieval-Augmented Generation for long documents
- Semantic chunking with ChromaDB vector store
- Multi-pass extraction with validation
- Best for: Verbose, unstructured PDFs (prospectuses, reports)

#### **3.2.4 Data Layer**

**Document Storage (S3/MinIO)**
- Original document persistence
- Versioning for audit trails
- Encryption at rest (AES-256)
- Retention policies for compliance

**Vector Database (ChromaDB)**
- Stores document embeddings for RAG
- Enables semantic search across document corpus
- Optimized for similarity queries

**Results Database (PostgreSQL)**
- Stores extracted entities with metadata
- Audit log of all extractions
- Supports complex queries for analytics

---

## 4. Integration Architecture

### 4.1 API Integration Patterns

#### **4.1.1 Synchronous API (Real-Time)**

```http
POST /api/v1/extract/sync
Content-Type: multipart/form-data
Authorization: Bearer <jwt_token>

{
  "file": <binary>,
  "document_type": "docx",
  "features": ["ner"],
  "priority": "high"
}

Response (200 OK):
{
  "request_id": "req_abc123",
  "entities": {
    "counterparty": "BANK ABC",
    "notional": "EUR 1 million",
    ...
  },
  "confidence_scores": {...},
  "processing_time_ms": 1250
}
```

**Use Case:** Interactive UI, small documents (<10 pages)

**SLA:** <5 seconds for DOCX, <60 seconds for PDF

#### **4.1.2 Asynchronous API (Batch)**

```http
POST /api/v1/extract/async
Content-Type: application/json
Authorization: Bearer <jwt_token>

{
  "document_url": "s3://bucket/document.pdf",
  "callback_url": "https://client.example.com/webhook",
  "features": ["ner", "summarization"]
}

Response (202 Accepted):
{
  "task_id": "task_xyz789",
  "status": "queued",
  "estimated_completion": "2024-12-16T12:30:00Z"
}

Callback (POST to client webhook):
{
  "task_id": "task_xyz789",
  "status": "completed",
  "result_url": "https://ador.cmi.com/results/xyz789"
}
```

**Use Case:** Batch processing, large documents, background jobs

**SLA:** <5 minutes for standard priority

### 4.2 Message Queue Architecture

```
┌─────────────┐       ┌──────────────┐       ┌─────────────┐
│  API Layer  │──────>│  RabbitMQ    │──────>│  Workers    │
│             │  Pub  │   Exchange   │  Sub  │  (3+ nodes) │
└─────────────┘       └──────────────┘       └─────────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
              ┌─────v──────┐    ┌────v───────┐
              │ high_prio  │    │ normal_prio│
              │  Queue     │    │  Queue     │
              └────────────┘    └────────────┘
```

**Queue Configuration:**
- **high_prio:** Max 100 messages, TTL 5 min
- **normal_prio:** Max 1000 messages, TTL 1 hour
- **dlq (dead letter):** Failed messages for investigation

---

## 5. Data Flow Diagrams

### 5.1 Document Upload Flow

```
┌─────────┐
│  User   │
└────┬────┘
     │
     │ 1. Upload document (Web UI)
     v
┌────────────┐
│   Web UI   │
└────┬───────┘
     │
     │ 2. POST /api/v1/extract
     v
┌────────────┐
│  API GW    │ 3. Authenticate (JWT)
└────┬───────┘    Authorize (RBAC)
     │
     │ 4. Store document
     v
┌────────────┐
│  S3/MinIO  │
└────┬───────┘
     │
     │ 5. Classify document type
     v
┌──────────────┐
│ Classifier   │
└────┬─────────┘
     │
     ├─── DOCX ──> Rule Parser ───┐
     │                             │
     ├─── TXT  ──> NER Extractor ─┤
     │                             │
     └─── PDF  ──> LLM + RAG ─────┤
                                   │
                                   v
                          ┌────────────────┐
                          │ Store Results  │
                          │  (PostgreSQL)  │
                          └────────┬───────┘
                                   │
                                   │ 6. Return results
                                   v
                             ┌───────────┐
                             │  User     │
                             └───────────┘
```

### 5.2 LLM + RAG Processing Flow

```
┌───────────┐
│ PDF Input │
└─────┬─────┘
      │
      │ 1. Extract text + tables
      v
┌────────────────┐
│  PDF Parser    │
│  (pdfplumber)  │
└──────┬─────────┘
       │
       │ 2. Semantic chunking
       v
┌────────────────┐
│  Chunker       │
│  (1000 tokens) │
└──────┬─────────┘
       │
       │ 3. Generate embeddings
       v
┌────────────────┐
│  Embed Model   │
│  (OpenAI)      │
└──────┬─────────┘
       │
       │ 4. Store in vector DB
       v
┌────────────────┐
│  ChromaDB      │
└──────┬─────────┘
       │
       │ 5. Query: "Find counterparty"
       v
┌────────────────┐
│  Retriever     │ ← Top-K chunks
└──────┬─────────┘
       │
       │ 6. Context + Prompt → LLM
       v
┌────────────────┐
│  Claude 3      │
│  Sonnet        │
└──────┬─────────┘
       │
       │ 7. Extract entities (JSON)
       v
┌────────────────┐
│  Validator     │ ← Confidence check
└──────┬─────────┘
       │
       │ 8. Store results
       v
┌────────────────┐
│  PostgreSQL    │
└────────────────┘
```

---

## 6. Non-Functional Requirements

### 6.1 Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| **API Latency (p95)** | <100ms | Prometheus metrics |
| **DOCX Processing** | <5 seconds | CloudWatch logs |
| **PDF Processing (RAG)** | <60 seconds | CloudWatch logs |
| **Throughput** | 1000 docs/day | Daily job metrics |
| **Concurrent Users** | 50 | Load testing |

### 6.2 Availability & Reliability

| Requirement | Target | Implementation |
|-------------|--------|----------------|
| **Uptime** | 99.5% (~43 min/month downtime) | Multi-AZ deployment |
| **RTO (Recovery Time)** | <30 minutes | Automated failover |
| **RPO (Recovery Point)** | <5 minutes | DB replication |
| **Error Rate** | <0.1% | Retry logic, circuit breakers |

### 6.3 Security

**Authentication:**
- JWT tokens with 1-hour expiration
- SSO integration with CMI AD/LDAP
- API keys for service-to-service communication

**Authorization:**
- Role-Based Access Control (RBAC)
- Roles: admin, analyst, read-only
- Document-level permissions

**Data Protection:**
- Encryption at rest (AES-256)
- Encryption in transit (TLS 1.3)
- PII detection and masking
- Audit logging (all API calls)

**Compliance:**
- GDPR: Right to erasure, data portability
- SOC 2: Access controls, audit trails
- ISO 27001: Information security management

### 6.4 Scalability

**Horizontal Scaling:**
- Stateless API layer (scale out via load balancer)
- Worker pool (add/remove based on queue depth)
- Database read replicas for analytics queries

**Vertical Scaling:**
- LLM inference on GPU instances (g4dn.xlarge)
- ChromaDB on memory-optimized instances (r5.2xlarge)

**Auto-Scaling Triggers:**
```yaml
scale_up_policy:
  metric: queue_depth
  threshold: >100 messages
  action: +2 workers

scale_down_policy:
  metric: queue_depth
  threshold: <10 messages
  cooldown: 10 minutes
  action: -1 worker
```

### 6.5 Monitoring & Observability

**Metrics (Prometheus + Grafana):**
- Request rate, latency, error rate (RED)
- Queue depth, processing time per document type
- LLM API costs, token usage
- Database query performance

**Logging (ELK Stack):**
- Structured JSON logs
- Correlation IDs for request tracing
- Error logs with stack traces

**Alerting (PagerDuty):**
- P0: API downtime, database failure
- P1: Error rate >1%, p95 latency >5s
- P2: Queue depth >500, extraction accuracy drop

**Tracing (Jaeger):**
- End-to-end request tracing
- Identify bottlenecks in multi-stage processing

---

## 7. Deployment Architecture

### 7.1 Infrastructure (AWS)

```
┌─────────────────────────────────────────────────────────────┐
│                      VPC (10.0.0.0/16)                      │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  Public Subnet (DMZ)                                  │ │
│  │  ┌──────────────┐         ┌──────────────┐          │ │
│  │  │ ALB          │         │ NAT Gateway  │          │ │
│  │  │ (API + UI)   │         │              │          │ │
│  │  └──────────────┘         └──────────────┘          │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  Private Subnet (Application)                         │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │ │
│  │  │ ECS Fargate  │  │ ECS Fargate  │  │ ECS Fargate│ │ │
│  │  │ (API Svc)    │  │ (Worker 1)   │  │ (Worker 2) │ │ │
│  │  └──────────────┘  └──────────────┘  └────────────┘ │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  Private Subnet (Data)                                │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │ │
│  │  │ RDS          │  │ ElastiCache  │  │ OpenSearch │ │ │
│  │  │ (PostgreSQL) │  │ (Redis)      │  │ (ChromaDB) │ │ │
│  │  └──────────────┘  └──────────────┘  └────────────┘ │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

External Services:
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  S3 Bucket   │  │  Secrets Mgr │  │  CloudWatch  │
│  (Documents) │  │  (API Keys)  │  │  (Logs)      │
└──────────────┘  └──────────────┘  └──────────────┘
```

### 7.2 Technology Stack

| Layer | Technology | Justification |
|-------|------------|---------------|
| **Frontend** | React + TypeScript | Modern, type-safe UI |
| **API** | FastAPI (Python 3.11) | High performance, OpenAPI |
| **Task Queue** | RabbitMQ | Reliable, AMQP standard |
| **Caching** | Redis | Fast, in-memory cache |
| **Database** | PostgreSQL 15 | Relational, ACID compliance |
| **Vector DB** | ChromaDB | Open-source, embeddings |
| **Storage** | S3 / MinIO | Object storage, versioning |
| **Container** | Docker + ECS Fargate | Serverless containers |
| **Monitoring** | Prometheus + Grafana | Metrics, dashboards |
| **Logging** | CloudWatch Logs | Centralized, searchable |

### 7.3 CI/CD Pipeline

```
┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
│  GitHub  │─────>│ GitHub   │─────>│  Build   │─────>│  Deploy  │
│  (Code)  │ Push │ Actions  │ Test │  (Docker)│ Push │  (ECS)   │
└──────────┘      └──────────┘      └──────────┘      └──────────┘
                       │                  │
                       v                  v
                  ┌──────────┐      ┌──────────┐
                  │  Unit    │      │  ECR     │
                  │  Tests   │      │ (Images) │
                  └──────────┘      └──────────┘
```

**Pipeline Stages:**
1. **Commit:** Developer pushes code to GitHub
2. **Test:** Unit tests, integration tests, linting
3. **Build:** Docker image build, tag with version
4. **Push:** Push to ECR (Elastic Container Registry)
5. **Deploy:** Update ECS service, rolling deployment

---

## 8. Disaster Recovery

### 8.1 Backup Strategy

| Component | Backup Frequency | Retention | Location |
|-----------|------------------|-----------|----------|
| **PostgreSQL** | Every 6 hours | 30 days | RDS automated backups |
| **S3 Documents** | Continuous (versioning) | 90 days | S3 Glacier |
| **Configuration** | On change | Indefinite | Git repository |
| **Vector DB** | Daily | 7 days | S3 snapshot |

### 8.2 Failover Procedures

**Database Failover:**
- RDS Multi-AZ automatic failover (<60 seconds)
- Read replica promotion for disaster recovery

**Application Failover:**
- ALB health checks (every 30 seconds)
- Unhealthy container replaced automatically
- Cross-region failover (manual trigger)

---

## 9. Migration & Rollout Plan

### Phase 1: Development (Week 1-4)
- [ ] Set up AWS infrastructure (VPC, subnets, security groups)
- [ ] Deploy PostgreSQL RDS instance
- [ ] Implement core API endpoints
- [ ] Integrate rule-based DOCX parser
- [ ] Deploy to dev environment

### Phase 2: Testing (Week 5-6)
- [ ] Integration testing with CMI IS
- [ ] Load testing (50 concurrent users)
- [ ] Security audit (penetration testing)
- [ ] User acceptance testing (UAT)

### Phase 3: Pilot (Week 7-8)
- [ ] Deploy to staging environment
- [ ] Onboard 10 pilot users
- [ ] Process 100 sample documents
- [ ] Collect feedback, iterate

### Phase 4: Production (Week 9-10)
- [ ] Deploy to production
- [ ] Enable monitoring/alerting
- [ ] Gradual rollout (10% → 50% → 100% traffic)
- [ ] Post-deployment validation

---

## 10. Risks & Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **LLM API downtime** | High | Low | Fallback to secondary LLM, queue requests |
| **Cost overrun** | Medium | Medium | Set budget alerts, optimize prompt length |
| **Data breach** | High | Low | Encryption, access control, audit logs |
| **Performance degradation** | Medium | Medium | Auto-scaling, caching, optimization |
| **Model accuracy drift** | High | Medium | A/B testing, continuous monitoring |

---

## 11. Conclusion

The ADOR system provides a scalable, secure, and intelligent solution for financial document processing. By leveraging modern AI techniques (rule-based, NER, LLM+RAG) and enterprise-grade architecture patterns, ADOR achieves high accuracy and performance while integrating seamlessly with the CMI Information System.

**Key Success Factors:**
1. Intelligent routing based on document type
2. Hybrid extraction approach (rules + ML + LLMs)
3. Production-ready infrastructure (monitoring, security, scalability)
4. Continuous improvement through monitoring and feedback loops

---

**Document Approval:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Solution Architect | Ali Mohamed | Dec 16, 2024 | [Digital Signature] |
| Technical Lead | [Pending] | [Pending] | [Pending] |
| Security Officer | [Pending] | [Pending] | [Pending] |
| Business Stakeholder | [Pending] | [Pending] | [Pending] |

---

**Revision History:**

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | Dec 16, 2024 | Initial document | Ali Mohamed |
| 0.9 | Dec 15, 2024 | Draft for review | Ali Mohamed |

---

**Appendices:**

- **Appendix A:** API Specification (OpenAPI/Swagger)
- **Appendix B:** Database Schema (ERD)
- **Appendix C:** Security Controls Matrix
- **Appendix D:** Cost Estimation ($10K/month operational costs)