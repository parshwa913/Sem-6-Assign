import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

const API_URL = 'http://localhost:3000/api';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  constructor(private http: HttpClient) { }

  // Authentication
  login(data: any): Observable<any> {
    return this.http.post(`${API_URL}/auth/login`, data);
  }

  // --- Admin endpoints ---
  addInstructor(instructor: any): Observable<any> {
    return this.http.post(`${API_URL}/admin/instructors`, instructor, this.getHeaders());
  }
  getInstructors(): Observable<any> {
    return this.http.get(`${API_URL}/admin/instructors`, this.getHeaders());
  }
  deleteInstructor(id: string): Observable<any> {
    return this.http.delete(`${API_URL}/admin/instructors/${id}`, this.getHeaders());
  }
  addStudent(student: any): Observable<any> {
    return this.http.post(`${API_URL}/admin/students`, student, this.getHeaders());
  }
  getStudents(): Observable<any> {
    return this.http.get(`${API_URL}/admin/students`, this.getHeaders());
  }
  deleteStudent(id: string): Observable<any> {
    return this.http.delete(`${API_URL}/admin/students/${id}`, this.getHeaders());
  }

  // --- Instructor endpoints ---
  addCourse(course: any): Observable<any> {
    return this.http.post(`${API_URL}/instructor/courses`, course, this.getHeaders());
  }
  getCourses(): Observable<any> {
    return this.http.get(`${API_URL}/instructor/courses`, this.getHeaders());
  }
  deleteCourse(id: string): Observable<any> {
    return this.http.delete(`${API_URL}/instructor/courses/${id}`, this.getHeaders());
  }

  // --- Student endpoint ---
  getStudentProfile(): Observable<any> {
    return this.http.get(`${API_URL}/student/profile`, this.getHeaders());
  }

  // Helper to add token header
  private getHeaders() {
    const token = localStorage.getItem('token') || '';
    return { headers: new HttpHeaders({ 'Authorization': `Bearer ${token}` }) };
  }
}
