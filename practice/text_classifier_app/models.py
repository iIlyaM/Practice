from django.db import models


class UserRecord(models.Model):
    user_login = models.CharField(max_length=200)
    text = models.TextField(max_length=10000)
    received_tags = models.CharField(max_length=100)



