import { Component, OnInit, ViewEncapsulation } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { interval } from 'rxjs';
import { switchMap } from 'rxjs/operators';
import { ApiService } from '../../services/api.service';
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-teacher-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './teacher-dashboard.component.html',
  styleUrls: ['./teacher-dashboard.component.css'],
  encapsulation: ViewEncapsulation.Emulated
})
export class TeacherDashboardComponent implements OnInit {
  examCount = 0;
  studentCount = 0;
  ongoingExams = 0;
  completedExams = 0;
  createdExams: any[] = [];
  reportData: any[] = [];
  teacherId!: number;
  currentTab: string = 'dashboard';

  newExam = {
    title: '',
    description: '',
    total_marks: 100,
    duration_minutes: 60,
    course_id: 1,
    created_by: this.teacherId,
    start_time: null,
    question_ids: JSON.stringify([]),
    assigned_students: JSON.stringify([]),
    status: 'upcoming'
  };

  questionOptions: string[] = ['option_a', 'option_b', 'option_c', 'option_d'];
  newQuestion: { [key: string]: any; course_id: number; question_text: string; option_a: string; option_b: string; option_c: string; option_d: string; correct_option: string; difficulty: string } = {
    course_id: 1,
    question_text: '',
    option_a: '',
    option_b: '',
    option_c: '',
    option_d: '',
    correct_option: 'A',
    difficulty: 'easy'
  };

  constructor(
    private api: ApiService,
    private authService: AuthService,
    private router: Router
  ) {
    const currentUser = this.authService.getCurrentUser();
    if (!currentUser || currentUser.userType !== 'teacher') {
      this.router.navigate(['/login']);
      return;
    }
    this.teacherId = currentUser.id;
  }

  ngOnInit(): void {
    this.loadDashboardData();
    interval(30000)
      .pipe(switchMap(() => this.api.getTeacherDashboard(this.teacherId)))
      .subscribe(
        data => this.updateDashboard(data),
        error => console.error('Error refreshing dashboard', error)
      );
  }

  setTab(tab: string): void {
    this.currentTab = tab;
    if (tab === 'reports') {
      this.loadReportData();
    } else {
      this.loadDashboardData();
    }
  }

  loadDashboardData(): void {
    // Provide dummy data that matches your database schema
    const dummyDashboardData = {
      examCount: 2,  
      studentCount: 2,  
      ongoingExams: 0, 
      completedExams: 0  
    };
    
    this.updateDashboard(dummyDashboardData);
    
    // These match the exam records in your database
    const dummyExams = [
      {
        id: 1,
        course_id: 1,
        created_by: 1,
        title: "Math Exam 1",
        description: "Covers basic math topics",
        total_marks: 100,
        start_time: "2025-03-01T10:00:00",
        duration_minutes: 60,
        question_ids: [1, 2, 3, 4],
        assigned_students: [1, 2],
        status: "upcoming",
        created_at: "2023-03-15T00:00:00"
      },
      {
        id: 2,
        course_id: 1,
        created_by: 1,
        title: "Math Exam 2",
        description: "Advanced math exam",
        total_marks: 100,
        start_time: "2025-03-15T10:00:00",
        duration_minutes: 90,
        question_ids: [5, 6, 7, 8],
        assigned_students: [1],
        status: "upcoming",
        created_at: "2023-03-15T00:00:00"
      }
    ];
    
    this.createdExams = dummyExams;
  }

  updateDashboard(data: any): void {
    this.examCount = data.examCount || 0;
    this.studentCount = data.studentCount || 0;
    this.ongoingExams = data.ongoingExams || 0;
    this.completedExams = data.completedExams || 0;
  }

  submitCreateExam(): void {
    try {
      // Use JSON.stringify for the JSON fields
      const examData = {
        course_id: Number(this.newExam.course_id),
        created_by: Number(this.teacherId),
        title: this.newExam.title,
        description: this.newExam.description || "", 
        total_marks: Number(this.newExam.total_marks),
        duration_minutes: Number(this.newExam.duration_minutes),
        start_time: this.newExam.start_time ? new Date(this.newExam.start_time).toISOString() : null,
        // Convert arrays to JSON strings for MySQL JSON columns
        question_ids: JSON.stringify([]),
        assigned_students: JSON.stringify([]),
        status: "upcoming"
      };

      console.log('Submitting Exam:', examData);
      this.api.createExam(examData).subscribe(
        response => {
          console.log('Exam created successfully:', response);
          alert('Exam created successfully.');
          this.createdExams.push(response.exam || response);
          this.loadDashboardData();
          // Reset form
          this.newExam = {
            title: '',
            description: '',
            total_marks: 100,
            duration_minutes: 60,
            course_id: 1,
            created_by: this.teacherId,
            start_time: null,
            question_ids: JSON.stringify([]),
            assigned_students: JSON.stringify([]),
            status: 'upcoming'
          };
        },
        error => {
          console.error('Error creating exam:', error);
          alert('Error creating exam: ' + (error.error?.message || error.statusText));
        }
      );
    } catch (e) {
      console.error('Error processing exam data:', e);
      alert('Error processing exam data: ' + e);
    }
  }

  submitManageQuestions(): void {
    const questionData = {
      created_by: this.teacherId,
      course_id: this.newQuestion.course_id,
      question_text: this.newQuestion.question_text,
      option_a: this.newQuestion.option_a,
      option_b: this.newQuestion.option_b,
      option_c: this.newQuestion.option_c,
      option_d: this.newQuestion.option_d,
      correct_option: this.newQuestion.correct_option,
      difficulty: this.newQuestion.difficulty
    };
    this.api.addQuestion(questionData).subscribe(
      response => {
        alert('Question added successfully.');
        this.newQuestion = {
          course_id: 1,
          question_text: '',
          option_a: '',
          option_b: '',
          option_c: '',
          option_d: '',
          correct_option: 'A',
          difficulty: 'easy'
        };
      },
      error => {
        console.error('Error adding question:', error);
        alert('Error adding question.');
      }
    );
  }

  submitGenerateReports(): void {
    this.api.getReportData().subscribe(
      data => {
        this.reportData = data;
      },
      error => {
        console.error('Error generating report:', error);
        alert('Error generating report.');
      }
    );
  }

  loadReportData(): void {
    this.submitGenerateReports();
  }

  logout(): void {
    const confirmLogout = confirm("Are you sure you want to logout?");
    if (confirmLogout) {
      this.authService.logout();
      this.router.navigate(['/login']);
    }
  }
}
