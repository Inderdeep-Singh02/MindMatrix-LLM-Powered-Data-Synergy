class IncludeAPIRouter(object):
    def __new__(cls):
        from application.main.routers.healthcheck import router as router_healthcheck
        from application.main.routers.analysis import router as analysis
        from application.main.routers.preprocess import router as preprocess
        from application.main.routers.model import router as model
        from fastapi.routing import APIRouter

        router = APIRouter()
        router.include_router(router_healthcheck, prefix='/api', tags=['health_check'])
        router.include_router(analysis, prefix='/api', tags=['analysis'])
        router.include_router(preprocess, prefix='/api', tags=['preprocess'] )
        router.include_router(model, prefix="/api", tags=['model'])

        return router

