FROM python:3.10-slim
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN apt update && apt install -y ffmpeg
RUN apt -y install wget
RUN apt -y install firefox-esr

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH \
    GECKODRIVERURL=https://github.com/mozilla/geckodriver/releases/download/v0.34.0/geckodriver-v0.34.0-linux64.tar.gz \
    GECKODRIVERFILENAME=geckodriver-v0.34.0-linux64.tar.gz
    

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

RUN wget -P $HOME/app $GECKODRIVERURL
RUN tar --warning=no-file-changed -xzf $HOME/app/$GECKODRIVERFILENAME
RUN rm $HOME/app/$GECKODRIVERFILENAME

RUN chmod +x geckodriver

RUN ls -ltr

EXPOSE 7860
ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "3"]