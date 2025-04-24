import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-instructor-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './instructor-dashboard.component.html',
  styleUrls: ['./instructor-dashboard.component.css']
})
export class InstructorDashboardComponent implements OnInit {
  instructor: any = {};         
  courses: any[] = [];         
  displayCourses: boolean = false;

  students: any[] = [];         
  newStudent: any = {         
    ID: '',
    name: '',
    dept_name: '',
    tot_cred: 0
  };
  isEditingStudent: boolean = false;
  studentMessage: string = '';

  constructor(private router: Router, private http: HttpClient) {}

  ngOnInit(): void {
    // Load instructor info
    const storedInstrName = localStorage.getItem('instructorName');
    console.log('Retrieved instructorName from localStorage:', storedInstrName);
    if (!storedInstrName) {
      console.error('Instructor name not found in localStorage');
      this.router.navigate(['/login']);
      return;
    }
    const instructorName = storedInstrName.trim();
    console.log('Instructor name after trim:', instructorName);
    this.loadInstructorInfoByName(instructorName);

    this.loadStudents();
  }


  loadInstructorInfoByName(name: string): void {
    const token = localStorage.getItem('token') || '';
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    const url = `http://localhost:3000/api/instructors/byName/${encodeURIComponent(name)}`;
    console.log('Requesting instructor info from URL:', url);
    this.http.get<any>(url, { headers }).subscribe(
      res => {
        console.log('Instructor info received:', res);
        this.instructor = res;
        if (!this.instructor || !this.instructor.ID) {
          window.alert('Instructor record not found or invalid.');
        }
      },
      err => {
        console.error('Error loading instructor info:', err);
        window.alert('Error loading instructor info: ' + (err.error?.message || err.message));
      }
    );
  }

  loadCourses(): void {
    const instructorId = this.instructor.ID;
    if (!instructorId) {
      window.alert('Instructor ID not found. Please log in again.');
      this.router.navigate(['/login']);
      return;
    }
    const token = localStorage.getItem('token') || '';
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    const url = `http://localhost:3000/api/instructors/${instructorId}/teaches`;
    console.log('Requesting courses info from URL:', url);
    this.http.get<any>(url, { headers }).subscribe(
      res => {
        console.log('Courses info received:', res);
        if (Array.isArray(res)) {
          this.courses = res;
        } else if (res && res.teaches) {
          this.courses = res.teaches;
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


  loadStudents(): void {
    const token = localStorage.getItem('token') || '';
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    const url = 'http://localhost:3000/api/students';
    console.log('Loading all students from:', url);
    this.http.get<any>(url, { headers }).subscribe(
      data => {
        console.log('Students loaded:', data);
        this.students = data;
      },
      err => {
        console.error('Error loading students:', err);
        this.studentMessage = err.error?.message || 'Error loading students';
      }
    );
  }

  addStudent(): void {
    if (!this.newStudent.ID || !this.newStudent.name || !this.newStudent.dept_name || this.newStudent.tot_cred === undefined) {
      this.studentMessage = 'Please fill in all required fields.';
      return;
    }
    const token = localStorage.getItem('token') || '';
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    const url = 'http://localhost:3000/api/students';
    console.log('Adding student with payload:', this.newStudent);
    this.http.post<any>(url, this.newStudent, { headers }).subscribe(
      res => {
        console.log('Student added:', res);
        this.studentMessage = 'Student added successfully.';
        this.loadStudents();
        this.clearStudentForm();
      },
      err => {
        console.error('Error adding student:', err);
        this.studentMessage = err.error?.message || 'Error adding student';
      }
    );
  }

  selectStudent(student: any): void {
    this.newStudent = { ...student }; 
    this.isEditingStudent = true;
    this.studentMessage = '';
  }

  updateStudent(): void {
    if (!this.newStudent.ID) {
      this.studentMessage = 'Student ID is required for update.';
      return;
    }
    const token = localStorage.getItem('token') || '';
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    const url = `http://localhost:3000/api/students/${this.newStudent.ID}`;
    console.log('Updating student with payload:', this.newStudent);
    this.http.put<any>(url, this.newStudent, { headers }).subscribe(
      res => {
        console.log('Student updated:', res);
        this.studentMessage = 'Student updated successfully.';
        this.loadStudents();
        this.clearStudentForm();
      },
      err => {
        console.error('Error updating student:', err);
        this.studentMessage = err.error?.message || 'Error updating student';
      }
    );
  }

  deleteStudent(ID: string): void {
    if (!confirm('Are you sure you want to delete this student?')) return;
    const token = localStorage.getItem('token') || '';
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    const url = `http://localhost:3000/api/students/${ID}`;
    console.log('Deleting student with ID:', ID);
    this.http.delete<any>(url, { headers }).subscribe(
      res => {
        console.log('Student deleted:', res);
        this.studentMessage = 'Student deleted successfully.';
        this.loadStudents();
      },
      err => {
        console.error('Error deleting student:', err);
        this.studentMessage = err.error?.message || 'Error deleting student';
      }
    );
  }

  clearStudentForm(): void {
    this.newStudent = { ID: '', name: '', dept_name: '', tot_cred: 0 };
    this.isEditingStudent = false;
    this.studentMessage = '';
  }

  // ---------- Logout ----------
  logout(): void {
    localStorage.clear();
    this.router.navigate(['/login'], { replaceUrl: true })
      .then(success => console.log('Logged out successfully:', success))
      .catch(err => {
        console.error('Navigation error during logout:', err);
        window.location.href = '/login';
      });
  }
}  
