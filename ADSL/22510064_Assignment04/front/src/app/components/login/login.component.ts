import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { ApiService } from '../../services/api.service';
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent {
  username = '';
  password = '';
  errorMessage = '';

  constructor(
    private api: ApiService,
    private router: Router,
    private authService: AuthService
  ) {}

  login() {
    this.errorMessage = '';
    const credentials = { username: this.username, password: this.password };

    this.api.login(credentials).subscribe({
      next: (response) => {
        console.log('Login response:', response);
        if (response && response.user) {
          // Save current user (make sure response.user contains the additional field "entityId")
          this.authService.setCurrentUser(response.user);
          // Navigate based on user type
          if (response.user.userType === 'teacher') {
            this.router.navigate(['/teacher-dashboard']);
          } else if (response.user.userType === 'student') {
            this.router.navigate(['/student-dashboard']);
          } else {
            this.errorMessage = 'Unknown user type';
          }
        } else {
          this.errorMessage = 'Login failed. Invalid server response.';
        }
      },
      error: (err) => {
        console.error('Login error:', err);
        this.errorMessage = err.error?.message || 'Login failed. Please try again.';
      }
    });
  }
}
