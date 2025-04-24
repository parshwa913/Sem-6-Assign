import { Component, OnInit } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { KeysPipe } from '../../pipes/keys.pipe';

@Component({
  selector: 'app-report-generator',
  standalone: true,
  imports: [CommonModule, FormsModule, KeysPipe],
  templateUrl: './report-generator.component.html',
  styleUrls: ['./report-generator.component.css']
})
export class ReportGeneratorComponent implements OnInit {
  selectedTable: string = 'student';
  allowedTables: string[] = ['student', 'instructor', 'department', 'course', 'classroom', 'section', 'teaches', 'takes', 'advisor', 'prereq', 'timeslot'];
  reportData: any[] = [];
  message: string = '';

  constructor(private http: HttpClient, private router: Router) {}

  ngOnInit(): void {}

  loadReport(): void {
    if (!this.selectedTable) {
      this.message = 'Please select a table.';
      return;
    }
    console.log('Selected table:', this.selectedTable);
    const token = localStorage.getItem('token') || '';
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    const url = `http://localhost:3000/api/report?table=${this.selectedTable}`;
    console.log('Requesting report from URL:', url);
    this.http.get<any>(url, { headers }).subscribe(
      data => {
        console.log('Report data:', data);
        this.reportData = data;
        if (this.reportData.length === 0) {
          this.message = 'No data found for the selected table.';
        } else {
          this.message = '';
        }
      },
      err => {
        console.error('Error loading report:', err);
        this.message = 'Error loading report: ' + (err.error?.message || err.message);
      }
    );
  }

  downloadCSV(): void {
    if (!this.reportData || this.reportData.length === 0) {
      this.message = 'No data to download.';
      return;
    }
    const csvData = this.convertToCSV(this.reportData);
    const blob = new Blob([csvData], { type: 'text/csv;charset=utf-8;' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${this.selectedTable}-report.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  }

  convertToCSV(objArray: any[]): string {
    if (objArray.length === 0) return '';
    const header = Object.keys(objArray[0]).join(',') + '\r\n';
    const rows = objArray.map(obj =>
      Object.values(obj).map(value => `"${value}"`).join(',')
    ).join('\r\n');
    return header + rows;
  }
}
