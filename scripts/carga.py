"""Manda 5 consultas de solo lectura a la vez y espera el estado final.

La API tiene que devolver los job_id enseguida. El tiempo de los agentes
aparece después, en GET /tasks/{id} y en Phoenix.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import httpx

CONSULTA = (
    "Un empleado va a cumplir 12 años en TechCorp. "
    "La política acredita vacaciones por tramos de antigüedad: "
    "los primeros 5 años, los 5 siguientes y los años por encima de 10. "
    "¿Cuántos días de vacaciones acumuló en total en esos 12 años? "
    "Necesito el dato de cada tramo según la política interna y el total "
    "(5 años del primer tramo, 5 del segundo y 2 del tercero)."
)

TERMINALES = {"DONE", "FAILED", "AWAITING_APPROVAL"}


async def main() -> None:
    parser = argparse.ArgumentParser(description="5 POST concurrentes contra la API")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    base = args.url.rstrip("/")

    async with httpx.AsyncClient(timeout=30) as client:
        inicio = time.perf_counter()
        respuestas = await asyncio.gather(
            *[client.post(f"{base}/tasks", json={"query": CONSULTA}) for _ in range(args.n)]
        )
        print(f"{args.n} POST respondieron en {time.perf_counter() - inicio:.2f}s")

        ids: list[str] = []
        for respuesta in respuestas:
            respuesta.raise_for_status()
            cuerpo = respuesta.json()
            print(cuerpo)
            if cuerpo.get("status") != "PENDING":
                sys.exit("El POST no devolvió PENDING")
            ids.append(cuerpo["job_id"])

        pendientes = set(ids)
        limite = time.perf_counter() + args.timeout
        while pendientes and time.perf_counter() < limite:
            await asyncio.sleep(2)
            for job_id in list(pendientes):
                estado = (await client.get(f"{base}/tasks/{job_id}")).json()
                if estado["status"] not in TERMINALES:
                    continue
                pendientes.discard(job_id)
                detalle = estado.get("result") or estado.get("error") or estado.get("interrupt")
                print(f"{job_id} {estado['status']} {detalle}")

    if pendientes:
        print("Siguieron abiertos:", ", ".join(pendientes))
        sys.exit(1)
    print("Listo. En Phoenix (http://localhost:6006), proyecto techcorp-api:")
    print("- trazas de estas ejecuciones")
    print("- costo por ejecución")
    print("- latencia p95")
    print("Guardá esas tres capturas en screenshots/.")


if __name__ == "__main__":
    asyncio.run(main())
