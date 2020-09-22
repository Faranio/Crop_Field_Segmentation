from lgblkb_tools import logger

from app.celery_tasks import salem


def main():
    for i in range(100):
        salem.delay(1, 2, 3, '123', qwe=dict(qwe='qweqwe'))

    pass


if __name__ == '__main__':
    main()
