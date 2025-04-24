// src/app/services/instructor.service.ts
import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { AuthService } from './auth.service';
import { Observable } from 'rxjs';

const API_URL = 'http://localhost:3000/api/instructors';

@Injectable({
  providedIn: 'root'
})
export class InstructorService {
  constructor(private http: HttpClient, private authService: AuthService) {}

  private getHeaders() {
    const token = this.authService.getToken();
    return new HttpHeaders({ 'Authorization': `Bearer ${token}` });
  }

  getInstructors(): Observable<any> {
    return this.http.get(API_URL, { headers: this.getHeaders() });
  }

  addInstructor(instructor: any): Observable<any> {
    return this.http.post(API_URL, instructor, { headers: this.getHeaders() });
  }

}
