class IncludeAPIRouter(object):
    def __new__(cls):
        from application.main.routers.healthcheck import router as router_healthcheck
        from application.main.routers.meta_visual import router as meta_visual
        from application.main.routers.meta_visual import router as upload_file
        from fastapi.routing import APIRouter

        router = APIRouter()
        router.include_router(router_healthcheck, prefix='/api', tags=['health_check'])
        router.include_router(meta_visual, prefix='/api', tags=['meta_visual'])

        return router

