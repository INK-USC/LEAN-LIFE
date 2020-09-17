psql -U postgres -c "REVOKE ALL ON DATABASE leanlife FROM leanlifeuser";
psql -U postgres -c "DROP DATABASE leanlife;"
psql -U postgres -c "DROP ROLE leanlifeuser";
psql -U postgres -c "CREATE DATABASE leanlife;"
psql -U postgres -c "CREATE USER leanlifeuser SUPERUSER";
psql -U postgres -c "ALTER ROLE leanlifeuser PASSWORD '$1'";
psql -U postgres -c "ALTER ROLE leanlifeuser SET client_encoding TO 'utf8';"
psql -U postgres -c "ALTER ROLE leanlifeuser SET default_transaction_isolation TO 'read committed';"
psql -U postgres -c "ALTER ROLE leanlifeuser SET timezone TO 'UTC';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE leanlife TO leanlifeuser;"
