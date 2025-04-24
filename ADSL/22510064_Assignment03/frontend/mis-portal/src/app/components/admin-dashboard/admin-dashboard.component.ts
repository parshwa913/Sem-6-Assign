import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-admin-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './admin-dashboard.component.html',
  styleUrls: ['./admin-dashboard.component.css']
})
export class AdminDashboardComponent implements OnInit {
  newInstructor = {
    ID: '',
    name: '',
    dept_name: '',
    salary: 0
  };

  instructors: any[] = [];
  displayInstructors: boolean = false;
  message: string = '';

  constructor(public router: Router, private http: HttpClient) {}

  ngOnInit(): void {}

  loadInstructors(): void {
    const token = localStorage.getItem('token') || '';
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    console.log("Attempting to load instructors...");
    this.http.get<any>('http://localhost:3000/api/instructors', { headers }).subscribe(
      response => {
        console.log("Instructors data received:", response);
        this.instructors = Array.isArray(response) ? response : response.instructors || [];
        this.displayInstructors = true;
      },
      error => {
        console.error("Error loading instructors:", error);
        this.message = 'Error loading instructors: ' + (error.error?.message || error.message);
      }
    );
  }

  addInstructor(): void {
    const token = localStorage.getItem('token') || '';
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    console.log('Adding instructor with payload:', this.newInstructor);
    this.http.post('http://localhost:3000/api/instructors', this.newInstructor, { headers }).subscribe(
      (response: any) => {
        console.log(response);
        this.message = 'Instructor added successfully';
        this.loadInstructors();
        this.newInstructor = { ID: '', name: '', dept_name: '', salary: 0 };
      },
      error => {
        console.error('Failed to add instructor:', error);
        this.message = 'Failed to add instructor: ' + (error.error?.message || '');
      }
    );
  }

  updateInstructor(instructor: any): void {
    const token = localStorage.getItem('token') || '';
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    const newName = prompt("Enter new name", instructor.name);
    const newDept = prompt("Enter new department", instructor.dept_name || instructor.department);
    const newSalary = prompt("Enter new salary", instructor.salary.toString());

    if (newName && newDept && newSalary) {
      const updatedInstructor = {
        name: newName,
        dept_name: newDept,
        salary: Number(newSalary)
      };

      this.http.put(`http://localhost:3000/api/instructors/${instructor.ID || instructor.id}`, updatedInstructor, { headers }).subscribe(
        (response: any) => {
          console.log(response);
          this.message = 'Instructor updated successfully';
          this.loadInstructors();
        },
        error => {
          console.error('Failed to update instructor:', error);
          this.message = 'Failed to update instructor: ' + (error.error?.message || '');
        }
      );
    }
  }

  deleteInstructor(id: string): void {
    const token = localStorage.getItem('token') || '';
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    if (confirm('Are you sure you want to delete this instructor?')) {
      this.http.delete(`http://localhost:3000/api/instructors/${id}`, { headers }).subscribe(
        (response: any) => {
          console.log(response);
          this.message = 'Instructor deleted successfully';
          this.loadInstructors();
        },
        error => {
          console.error('Failed to delete instructor:', error);
          this.message = 'Failed to delete instructor: ' + (error.error?.message || '');
        }
      );
    }
  }

  
  goToReport(): void {
    console.log('Navigating to report generator...');
    this.router.navigateByUrl('/report-generator')
      .then(success => console.log('Navigation to report generator successful:', success))
      .catch(err => console.error('Navigation error in goToReport():', err));
  }

  logout(): void {
    console.log('Logging out...');
    localStorage.clear();
    this.router.navigate(['/login'], { replaceUrl: true })
      .then(success => console.log('Navigation to login successful:', success))
      .catch(err => {
        console.error('Navigation error:', err);
        window.location.href = '/login';
      });
  }

}
