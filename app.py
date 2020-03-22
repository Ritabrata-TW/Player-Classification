from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles
from starlette.responses import JSONResponse
from starlette.templating import Jinja2Templates
from fastai.vision import (
    open_image,
    load_learner
)

from io import BytesIO
import uvicorn
import aiohttp


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


templates = Jinja2Templates(directory='templates')

app = Starlette(debug=True)
app.mount('/static', StaticFiles(directory='statics'), name='static')

learn = load_learner('./')


@app.route('/', methods=["GET"])
async def homepage(request):
    template = "index.html"
    context = {"request": request}
    if "image-url" in request.query_params:
        if len(request.query_params["image-url"]) > 0:
            bytes = await get_bytes(request.query_params["image-url"])
            img = open_image(BytesIO(bytes))
            _, _, losses = learn.predict(img)

            sorted_classes = sorted(
                zip(learn.data.classes, map(float, losses)),
                key=lambda p: p[1],
                reverse=True)

            context = {"request": request, "result": sorted_classes[0][0]}
    return templates.TemplateResponse(template, context)


@app.route('/error')
async def error(request):
    """
    An example error. Switch the `debug` setting to see either tracebacks or 500 pages.
    """
    raise RuntimeError("Oh no")


# @app.route("/upload", methods=["POST"])
# async def upload(request):
#     data = await request.form()
#     bytes = await (data["file"].read())
#     return predict_image_from_bytes(bytes)

@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)


def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))

    _, _, losses = learn.predict(img)
    return JSONResponse({
        "predictions": sorted(
            zip(learn.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    })


@app.exception_handler(404)
async def not_found(request, exc):
    """
    Return an HTTP 404 page.
    """
    template = "404.html"
    context = {"request": request}
    return templates.TemplateResponse(template, context, status_code=404)


@app.exception_handler(500)
async def server_error(request, exc):
    """
    Return an HTTP 500 page.
    """
    template = "500.html"
    context = {"request": request}
    return templates.TemplateResponse(template, context, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
