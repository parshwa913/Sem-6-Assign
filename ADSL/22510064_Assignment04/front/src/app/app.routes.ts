// src/app/app.routes.ts
import { Routes } from '@angular/router';
import { LoginComponent } from './components/login/login.component';
import { TeacherDashboardComponent } from './components/teacher-dashboard/teacher-dashboard.component';
import { StudentDashboardComponent } from './components/student-dashboard/student-dashboard.component';
import { ExamComponent } from './components/exam/exam.component';
import { QuestionBankComponent } from './components/question-bank/question-bank.component';
import { ReportComponent } from './components/report/report.component';
import { AppComponent } from './app.component';

export const APP_ROUTES: Routes = [
  { path: '', component: AppComponent },
  // Add more routes as needed
];

export const routes: Routes = [
  { path: '', redirectTo: 'login', pathMatch: 'full' },
  { path: 'login', component: LoginComponent },
  { path: 'teacher-dashboard', component: TeacherDashboardComponent },
  { path: 'student-dashboard', component: StudentDashboardComponent },
  { path: 'exam/:id', component: ExamComponent },
  { path: 'question-bank', component: QuestionBankComponent },
  { path: 'report', component: ReportComponent }
];
