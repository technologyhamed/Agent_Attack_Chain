# 1. ایجاد دایرکتوری‌های مورد نیاز
sudo mkdir -p /opt/gai_neo4j/{data,logs,import,plugins,config}
sudo chown -R 7474:7474 /opt/gai_neo4j/data  # Neo4j runs as user 7474

# 2. دانلود پلاگین‌های ضروری (یک‌بار)
cd /opt/gai_neo4j/plugins
# wget https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/25.7.0/apoc-25.7.0.jar
# wget https://github.com/neo4j/graph-data-science/releases/download/25.7.0/neo4j-graph-data-science-25.7.0.jar
# wget https://github.com/neo4j-labs/neosemantics/releases/download/5.18.0/neosemantics-5.18.0.jar

# 3. راه‌اندازی سرویس‌ها
cd /opt/gai_neo4j
docker compose -f docker-compose.yml up -d

# 4. بررسی سلامت سرویس‌ها
docker compose ps
# باید هر سه سرویس در وضعیت "Up (healthy)" باشند

# 5. دسترسی به سرویس‌ها
# Neo4j Browser: https://185.130.79.32:7473 
# NeoDash: http://185.130.79.32:8080
# MCP Server: http://185.130.79.32:8000