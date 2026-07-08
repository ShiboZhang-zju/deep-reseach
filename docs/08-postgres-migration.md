# 08 - PostgreSQL 迁移

## 问题
- SQLite 不支持并发写入，多任务同时运行会 `database is locked`
- busy_timeout 30s 只是缓解，不是根本解决
- 随着任务增多，锁冲突会更频繁

## 方案

### 1. 添加 PostgreSQL 支持
- `config.py` database_url 支持 postgresql://
- `session.py` 根据数据库类型调整连接参数
- SQLAlchemy ORM 兼容两种数据库

### 2. Docker Compose
```yaml
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_DB: deep_research
      POSTGRES_USER: research
      POSTGRES_PASSWORD: research
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
```

### 3. 迁移脚本
SQLite → PostgreSQL 数据迁移脚本，保留已有任务和论文数据。

## 涉及文件
- `backend/app/config.py` — database_url
- `backend/app/db/session.py` — 连接参数
- `docker-compose.yml` — 新建
- `backend/scripts/migrate_sqlite_to_pg.py` — 新建

## 验证
- 两个任务同时运行无锁冲突
- 已有数据完整迁移
