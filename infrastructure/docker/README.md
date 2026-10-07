# PostgreSQL con pgvector

Esta configuración inicia PostgreSQL 16 con la extensión `pgvector` y conserva los datos en el volumen Docker nombrado `restaurant-ai-postgres-data`.

## Arranque

1. Copia el archivo de variables y define una contraseña segura:

   ```powershell
   Copy-Item .env.example .env
   ```

2. Desde esta carpeta, inicia la base de datos:

   ```powershell
   docker compose up -d
   ```

3. Comprueba que el servicio esté listo:

   ```powershell
   docker compose ps
   ```

La cadena de conexión por defecto es:

```text
postgresql://restaurant_ai:<POSTGRES_PASSWORD>@localhost:5432/restaurant_ai
```

El script `init-db.sql` habilita `vector` automáticamente al crear un volumen nuevo. Puedes verificarlo con:

```powershell
docker compose exec postgres psql -U restaurant_ai -d restaurant_ai -c "\dx"
```

## Persistencia y mantenimiento

Los datos permanecen al detener o recrear el contenedor porque Docker los guarda en el volumen `restaurant-ai-postgres-data`.

```powershell
docker compose down
docker compose up -d
```

No ejecutes `docker compose down --volumes` salvo que quieras eliminar permanentemente la base de datos. Para inspeccionar los volúmenes:

```powershell
docker volume ls
```
