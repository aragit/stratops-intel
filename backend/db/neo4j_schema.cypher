// Constraints
CREATE CONSTRAINT company_name IF NOT EXISTS
FOR (c:Company) REQUIRE c.name IS UNIQUE;

CREATE CONSTRAINT person_email IF NOT EXISTS
FOR (p:Person) REQUIRE p.email IS UNIQUE;

CREATE CONSTRAINT product_id IF NOT EXISTS
FOR (pr:Product) REQUIRE pr.id IS UNIQUE;

// Indexes
CREATE INDEX priced_at_date IF NOT EXISTS
FOR ()-[r:PRICED_AT]-() ON (r.valid_from, r.valid_to);

CREATE INDEX employed_at_date IF NOT EXISTS
FOR ()-[r:EMPLOYED_AT]-() ON (r.valid_from, r.valid_to);

CREATE INDEX mentioned_in_date IF NOT EXISTS
FOR ()-[r:MENTIONED_IN]-() ON (r.valid_from, r.valid_to);

// Tenant isolation index on all nodes
CREATE INDEX tenant_id_company IF NOT EXISTS
FOR (c:Company) ON (c.tenant_id);

CREATE INDEX tenant_id_person IF NOT EXISTS
FOR (p:Person) ON (p.tenant_id);

// Node type definitions
CREATE (c:Company {name: $name, ticker: $ticker, industry: $industry, founded_date: $founded_date, tenant_id: $tenant_id})
MERGE (c:Company {name: $name, tenant_id: $tenant_id})
ON SET c.ticker = $ticker, c.industry = $industry, c.founded_date = $founded_date;

CREATE (p:Person {name: $name, email: $email, title: $title, tenant_id: $tenant_id})
MERGE (p:Person {name: $name, email: $email, tenant_id: $tenant_id})
ON SET p.title = $title;

CREATE (pr:Product {id: $id, name: $name, category: $category, tenant_id: $tenant_id})
MERGE (pr:Product {id: $id, tenant_id: $tenant_id})
ON SET pr.name = $name, pr.category = $category;

CREATE (s:Signal {id: $id, source_type: $source_type, source_url: $source_url, fingerprint: $fingerprint, tenant_id: $tenant_id})
MERGE (s:Signal {id: $id, tenant_id: $tenant_id})
ON SET s.source_type = $source_type, s.source_url = $source_url, s.fingerprint = $fingerprint;

// Relationship types
// (Company)-[:COMPETES_WITH {strength, valid_from, valid_to}]->(Company)
// (Company)-[:PRICED_AT {price, currency, valid_from, valid_to}]->(Product)
// (Person)-[:EMPLOYED_AT {role, seniority, valid_from, valid_to}]->(Company)
// (Person)-[:MENTIONED_IN {sentiment, valid_from, valid_to}]->(Signal)
// (Company)-[:MENTIONED_IN {sentiment, valid_from, valid_to}]->(Signal)
MATCH (c1:Company {tenant_id: $tenant_id}), (c2:Company {tenant_id: $tenant_id})
WHERE c1.name < c2.name
MERGE (c1)-[r:COMPETES_WITH {strength: $strength, valid_from: $valid_from, valid_to: $valid_to}]->(c2)
ON SET r.strength = $strength, r.valid_from = $valid_from, r.valid_to = $valid_to;

MATCH (c:Company {tenant_id: $tenant_id}), (pr:Product {tenant_id: $tenant_id})
MERGE (c)-[r:PRICED_AT {price: $price, currency: $currency, valid_from: $valid_from, valid_to: $valid_to}]->(pr)
ON SET r.price = $price, r.currency = $currency, r.valid_from = $valid_from, r.valid_to = $valid_to;

MATCH (p:Person {tenant_id: $tenant_id}), (c:Company {tenant_id: $tenant_id})
MERGE (p)-[r:EMPLOYED_AT {role: $role, seniority: $seniority, valid_from: $valid_from, valid_to: $valid_to}]->(c)
ON SET r.role = $role, r.seniority = $seniority, r.valid_from = $valid_from, r.valid_to = $valid_to;

MATCH (p:Person {tenant_id: $tenant_id}), (s:Signal {tenant_id: $tenant_id})
MERGE (p)-[r:MENTIONED_IN {sentiment: $sentiment, valid_from: $valid_from, valid_to: $valid_to}]->(s)
ON SET r.sentiment = $sentiment, r.valid_from = $valid_from, r.valid_to = $valid_to;

MATCH (c:Company {tenant_id: $tenant_id}), (s:Signal {tenant_id: $tenant_id})
MERGE (c)-[r:MENTIONED_IN {sentiment: $sentiment, valid_from: $valid_from, valid_to: $valid_to}]->(s)
ON SET r.sentiment = $sentiment, r.valid_from = $valid_from, r.valid_to = $valid_to;