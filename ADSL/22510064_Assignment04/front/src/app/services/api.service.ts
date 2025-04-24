import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';

@Injectable({
  providedIn: 'root',
})
export class ApiService {
  private baseUrl = 'http://localhost:3000'; // match your backend

  constructor(private http: HttpClient) {}

  // Login - KEEP AS IS since it works
  login(credentials: any): Observable<any> {
    return this.http.post(`${this.baseUrl}/users/login`, credentials);
  }

  // Teacher endpoints - UPDATED TO MATCH YOUR SERVER.JS ROUTES
  getTeacherDashboard(teacherId: number): Observable<any> {
    // Changed to match /dashboard endpoint in server.js
    return this.http.get<any>(`${this.baseUrl}/dashboard/teacher/${teacherId}`);
  }

  getRecentExams(teacherId: number): Observable<any[]> {
    // Changed to match /exams endpoint in server.js
    return this.http.get<any[]>(`${this.baseUrl}/exams?teacherId=${teacherId}`);
  }

  createExam(examData: any): Observable<any> {
    // Changed to match /exams endpoint in server.js
    return this.http.post<any>(`${this.baseUrl}/exams`, examData);
  }

  // View exam in progress
  assignExam(assignData: any): Observable<any> {
    // Matches /assign-exam endpoint in server.js
    return this.http.post<any>(`${this.baseUrl}/assign-exam`, assignData);
  }

  addQuestion(questionData: any): Observable<any> {
    // Matches /questions endpoint in server.js
    return this.http.post<any>(`${this.baseUrl}/questions`, questionData);
  }

  getReportData(): Observable<any[]> {
    // Matches /report endpoint in server.js
    return this.http.get<any[]>(`${this.baseUrl}/report`);
  }

  getTeacherNotifications(teacherId: number): Observable<any[]> {
    // Adjusted for /dashboard endpoint structure
    return this.http.get<any[]>(`${this.baseUrl}/dashboard/notifications/teacher/${teacherId}`);
  }

  // Student endpoints - adjusted to match server.js
  getStudentExams(studentId: number): Observable<any[]> {
    console.log(`Calling API endpoint: ${this.baseUrl}/student/${studentId}/exams`);
    return this.http.get<any[]>(`${this.baseUrl}/student/${studentId}/exams`)
      .pipe(
        catchError(error => {
          console.error('API error in getStudentExams:', error);
          // Return empty array as fallback
          return of([]);
        })
      );
  }

  submitExam(submissionData: any): Observable<any> {
    return this.http.post<any>(`${this.baseUrl}/student/submit-exam`, submissionData);
  }

  getUpcomingExams(studentId: number): Observable<any[]> {
    return this.http.get<any[]>(`${this.baseUrl}/student/${studentId}/exams/upcoming`);
  }

  getPerformanceStats(studentId: number): Observable<any> {
    return this.http.get<any>(`${this.baseUrl}/student/${studentId}/performance`);
  }

  getStudentNotifications(studentId: number): Observable<any[]> {
    return this.http.get<any[]>(`${this.baseUrl}/notifications/student/${studentId}`);
  }

  getExamById(id: number): Observable<any> {
    return this.http.get<any>(`${this.baseUrl}/exams/${id}`);
  }

  getQuestionsByIds(ids: number[]): Observable<any[]> {
    return this.http.post<any[]>(`${this.baseUrl}/questions/byIds`, { ids });
  }

  getExamResults(studentId: number): Observable<any[]> {
    return this.http.get<any[]>(`${this.baseUrl}/student/${studentId}/exam-results`);
  }

  createExamAttempt(attemptData: any): Observable<any> {
    return this.http.post<any>(`${this.baseUrl}/student/exam-attempts`, attemptData);
  }

  saveAnswer(answerData: any): Observable<any> {
    return this.http.post<any>(`${this.baseUrl}/student/answers`, answerData);
  }
}