from django.shortcuts import render, redirect
from django.views.generic import ListView, DetailView, CreateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from .forms import *
from django.contrib.auth.views import LoginView
from django.contrib.auth import logout, login


def home(request):
    return render(request, 'text_classifier_app/home.html')


class RegisterUser(CreateView):
    form_class = RegisterUserForm
    template_name = 'text_classifier_app/register.html'
    success_url = reverse_lazy('login')

    def form_valid(self, form):
        user = form.save()
        login(self.request, user)
        return redirect('home')


class LoginUser(LoginView):
    form_class = LoginUserForm
    template_name = 'text_classifier_app/login.html'

    def get_success_url(self):
        return reverse_lazy('home')


def logout_user(request):
    logout(request)
    return redirect('login')


class AddRecord(LoginRequiredMixin, CreateView):
    form_class = RecordForm
    template_name = 'text_classifier_app/addrecord.html'

    # def get_context_data(self, *, object_list=None, **kwargs):
    #     context = super().get_context_data(**kwargs)
    #     return dict(list(context.items()))