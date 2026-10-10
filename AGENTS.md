# Instrucciones para agentes de código — Restaurant AI RAG

## Objetivo
Este repositorio es un proyecto funcional y un laboratorio de aprendizaje avanzado de AI Engineering. Consulta ` docs/ROADMAP.md` para el plan maestro y el estado orientativo de las fases.

## Forma de trabajar
1. Inspecciona el código, dependencias, pruebas y documentación pertinentes antes de cambiar nada.
2. Para tareas amplias, presenta un plan breve con archivos afectados, dependencias y riesgos. Si el usuario pide aprobación previa, espera antes de implementar.
3. Trabaja en cambios pequeños y enfocados. No reescribas archivos o subsistemas no relacionados.
4. Comprueba `git status` y el diff; nunca asumas que el árbol está limpio.
5. Conserva los cambios existentes. No hagas commit ni push salvo petición explícita.
6. Ejecuta las pruebas pertinentes y reporta exactamente qué comandos ejecutaste y sus resultados. No afirmes que una prueba pasó si no se ejecutó.
7. Distingue pruebas unitarias, integración y validaciones que llamen a proveedores externos o puedan generar costes.
8. Al terminar, resume cambios, pruebas, limitaciones y próximos pasos.

## Objetivo educativo
No descartes una tecnología incluida en ` docs/ROADMAP.md` solo porque la implementación actual funcione sin ella. El objetivo es aprender e integrar las tecnologías progresivamente sin perder calidad técnica. Explica los conceptos nuevos, las decisiones, las alternativas y cómo se defenderían en una entrevista.

## Arquitectura
- Mantén la arquitectura modular existente y aplica DDD pragmáticamente; no introduzcas capas, repositorios o agregados sin justificación.
- Mantén la dirección hexagonal: el dominio no depende de frameworks, ORM, proveedores ni interfaces; la aplicación contiene casos de uso y define los puertos que necesita; infraestructura los implementa e interfaces adaptan sus protocolos.
- Reutiliza servicios de aplicación y Tools existentes en lugar de duplicar lógica.
- Mantén separados endpoints, Agent, Workflows y servidor MCP salvo requisito explícito.
- No expongas IDs internos, rutas locales, secretos ni detalles de infraestructura a modelos o clientes sin necesidad.
- Maneja errores y resultados vacíos de forma explícita.
- No añadas dependencias sin justificar necesidad y compatibilidad.

## Nombres y cambios de módulos
- Usa `snake_case` para los módulos Python y el lenguaje ubicuo del bounded context para los nombres.
- Nombra los casos de uso con acciones, los puertos con la capacidad requerida y los adaptadores con su responsabilidad y tecnología cuando aporte claridad.
- Distingue requests, responses y DTO de modelos ORM y de conceptos del dominio. No añadas sufijos como `_service`, `_manager`, `_handler` o `_repository` por defecto.
- Evita nombres genéricos si ocultan la responsabilidad; conserva excepciones idiomáticas o claramente acotadas por su paquete.
- Antes de crear, mover, dividir, renombrar o eliminar un módulo, inspecciona su responsabilidad, equivalentes, referencias y contratos; actualiza tests, configuración y documentación afectados. No renombres solo por estética.
- Aplica el detalle, ejemplos y excepciones de [las convenciones de nombres y módulos](backend/ARCHITECTURE.md#naming-and-module-placement).

## Recuperación y datos
No modifiques sin autorización expresa:
- modelos o dimensiones de embeddings;
- preprocesamiento de texto o imágenes;
- corpus;
- ranking, filtros de relevancia o umbrales;
- contratos públicos de Tools y endpoints;
- datos y etiquetas de evaluación.

No inventes etiquetas, métricas ni verificaciones manuales.

## Tecnologías y secuencia
- Sigue ` docs/ROADMAP.md`.
- Verifica el código y el roadmap antes de asumir si LangChain ya está integrado; amplía su uso en RAG de forma incremental y sin alterar recuperación o contratos protegidos.
- LangGraph se utiliza para orquestación con estado y transiciones cuando corresponda.
- MCP tiene un servidor local stdio según el último estado reportado; inspecciona antes de modificarlo.
- A2A ya está implementado para colaboración de planificación y búsqueda; inspecciona su contrato y mantén separados A2A, LangGraph y MCP. No añadas agentes o flujos de colaboración sin un caso justificado.
- Los benchmarks reales están aplazados hasta que el usuario decida retomarlos.

## Documentación
Cuando introduzcas una tecnología, documenta propósito, conceptos clave, ejecución, configuración, pruebas, limitaciones y alternativas consideradas.
