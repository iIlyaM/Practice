from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.forms import ModelForm
from .models import UserRecord
from django.contrib.auth.models import User
import pickle
import os

from .services.classifier import get_top_k_predictions


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
    # input_text = forms.CharField(widget=forms.Textarea(attrs={'cols': 50, 'rows': 10}))

    class Meta:
        model = UserRecord
        fields = ('input_text', 'user_login',)
        widgets = {
            'input_text': forms.Textarea(attrs={'cols': 50, 'rows': 10}),
            'user_login': forms.TextInput(),
        }

    def save(self, commit=True):
        record = super(RecordForm, self).save(commit)
        text = self.cleaned_data['input_text']

        model_path = "text_classifier_app/services/model.pkl"
        print(os.path.exists(model_path))
        transformer_path = "text_classifier_app/services/transformer.pkl"

        loaded_model = pickle.load(open(model_path, 'rb'))
        loaded_transformer = pickle.load(open(transformer_path, 'rb'))

        test_features = loaded_transformer.transform([text])
        tags = get_top_k_predictions(loaded_model, test_features, 2)[0]
        tags = ', '.join(tags)
        print(record.input_text)

        record.received_tags = tags
        record.save()
        return record


# class TagsForm(ModelForm):
#     class Meta:
#         model = UserRecord
#         fields = ('received_tags',)