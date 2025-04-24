import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-report',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './report.component.html',
  styleUrls: ['./report.component.css']
})

export class ReportComponent implements OnInit {
  reportData: any[] = [];

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.api.getReportData().subscribe((data: any[]) => {
      this.reportData = data;
    });
  }
}
