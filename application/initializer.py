class IncludeAPIRouter(object):
    def __new__(cls):
        from application.main.routers.healthcheck import router as router_healthcheck
        from application.main.routers.EdaVis import router as EdaVis
        from application.main.routers.EdaVis import router as create_upload_file
        from fastapi.routing import APIRouter

        router = APIRouter()
        router.include_router(router_healthcheck, prefix='/api', tags=['health_check'])
        router.include_router(create_upload_file, prefix='/api', tags=['upload'])
        router.include_router(EdaVis, prefix='/api', tags=['EdaVis'])
        
        return router

