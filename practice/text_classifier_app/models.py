from django.db import models


class UserRecord(models.Model):
    user_login = models.CharField(max_length=200)
    input_text = models.TextField(max_length=10000)
    received_tags = models.CharField(max_length=100)

    def get_absolute_url(self):
        return 'http://127.0.0.1:8000/'

