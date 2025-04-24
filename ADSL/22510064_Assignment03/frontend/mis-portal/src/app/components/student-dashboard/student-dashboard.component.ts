import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-student-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './student-dashboard.component.html',
  styleUrls: ['./student-dashboard.component.css']
})
export class StudentDashboardComponent implements OnInit {
  student: any = {};      // Student profile information
  advisor: any = {};      // Advisor information
  courses: any[] = [];    // Courses taken by the student
  displayCourses: boolean = false;

  constructor(private router: Router, private http: HttpClient) {}

  ngOnInit(): void {
    // Assume that the studentName is stored in localStorage after login
    const storedName = localStorage.getItem('studentName');
    console.log('Retrieved studentName from localStorage:', storedName);
    if (!storedName) {
      console.error('Student name not found in localStorage');
      this.router.navigate(['/login']);
      return;
    }
    const studentName = storedName.trim();
    console.log('Student name after trim:', studentName);
    this.loadStudentInfoByName(studentName);
  }

  loadStudentInfoByName(name: string): void {
    const token = localStorage.getItem('token') || '';
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    const url = `http://localhost:3000/api/students/byName/${encodeURIComponent(name)}`;
    console.log('Requesting student info from URL:', url);
    this.http.get<any>(url, { headers }).subscribe(
      res => {
        console.log('Student info received:', res);
        this.student = res;
        if (!this.student || !this.student.ID) {
          window.alert('Student record not found or invalid.');
        } else {
          this.loadAdvisor(this.student.ID);
        }
      },
      err => {
        console.error('Error loading student info:', err);
        window.alert('Error loading student info: ' + (err.error?.message || err.message));
      }
    );
  }

  loadAdvisor(studentId: string): void {
    const token = localStorage.getItem('token') || '';
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    const url = `http://localhost:3000/api/students/${studentId}/advisor`;
    console.log('Requesting advisor info from URL:', url);
    this.http.get<any>(url, { headers }).subscribe(
      res => {
        console.log('Advisor info received:', res);
        this.advisor = res;
      },
      err => {
        console.error('Error loading advisor info:', err);
        window.alert('Error loading advisor info: ' + (err.error?.message || err.message));
      }
    );
  }

  loadCourses(): void {
    const studentId = this.student.ID;
    if (!studentId) {
      window.alert('Student ID not found. Please log in again.');
      this.router.navigate(['/login']);
      return;
    }
    const token = localStorage.getItem('token') || '';
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    const url = `http://localhost:3000/api/students/${studentId}/takes`;
    console.log('Requesting courses info from URL:', url);
    this.http.get<any>(url, { headers }).subscribe(
      res => {
        console.log('Courses info received:', res);
        if (Array.isArray(res)) {
          this.courses = res;
        } else if (res && res.takes) {
          this.courses = res.takes;
        } else {
          this.courses = [];
        }
        this.displayCourses = true;
      },
      err => {
        console.error('Error loading courses:', err);
        window.alert('Error loading courses: ' + (err.error?.message || err.message));
      }
    );
  }

  logout(): void {
    // Clear local storage and force a full page reload to login
    localStorage.removeItem('token');
    localStorage.removeItem('studentName');
    window.location.href = '/login';
  }
}
