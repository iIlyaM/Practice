from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.forms import ModelForm
from .models import UserRecord
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError


class RegisterUserForm(UserCreationForm):
    username = forms.CharField(label='Login', widget=forms.TextInput(attrs={'class': 'form-input'}))
    email = forms.EmailField(label='Email', widget=forms.EmailInput(attrs={'class': 'form-input'}))
    password1 = forms.CharField(label='Password', widget=forms.PasswordInput(attrs={'class': 'form-input'}))
    password2 = forms.CharField(label='Password_repeat', widget=forms.PasswordInput(attrs={'class': 'form-input'}))

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')


class LoginUserForm(AuthenticationForm):
    username = forms.CharField(label='Login', widget=forms.TextInput(attrs={'class': 'form-input'}))
    password = forms.CharField(label='Password', widget=forms.PasswordInput(attrs={'class': 'form-input'}))


class RecordForm(ModelForm):
    # username = forms.CharField()
    # received_tags = forms.CharField(min_length=0)
    # text = forms.CharField(widget=forms.Textarea(attrs={'cols': 50, 'rows': 10}))

    class Meta:
        model = UserRecord
        fields = ('text',)
        widgets = {
            'text': forms.Textarea(attrs={'cols': 50, 'rows': 10})
        }
