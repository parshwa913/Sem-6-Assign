import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { AuthService } from '../../services/auth.service';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

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
  role = ''; // Expect values: 'admin', 'instructor', or 'student'
  errorMessage = '';

  constructor(private authService: AuthService, private router: Router) {}

  onLogin(): void {
    console.log('Login attempted:', this.username, this.role);
    this.authService.login(this.username, this.password, this.role).subscribe(
      data => {
        this.authService.setSession(data);
        if (this.role === 'admin') {
          // For admin, we expect an "admin_id"
          localStorage.setItem('adminId', data.user.admin_id);
          this.router.navigate(['/admin-dashboard']);
        } else if (this.role === 'instructor') {
          // For instructors, store the username (which holds the instructor's name)
          console.log('Storing instructor name:', data.user.username);
          localStorage.setItem('instructorName', data.user.username);
          this.router.navigate(['/instructor-dashboard']);
        } else if (this.role === 'student') {
          // For students, store the username (which holds the student's name)
          console.log('Storing student name:', data.user.username);
          localStorage.setItem('studentName', data.user.username);
          this.router.navigate(['/student-dashboard']);
        }
      },
      err => {
        this.errorMessage = err.error.message || 'Login failed';
        console.error('Login error:', err);
      }
    );
  }
}
