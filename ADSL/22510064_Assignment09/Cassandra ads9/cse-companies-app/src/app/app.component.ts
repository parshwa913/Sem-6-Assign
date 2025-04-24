import { Component } from '@angular/core';
import { CompanyListComponent } from './company-list.component';
import { CompanyFormComponent } from './company-form.component';

@Component({
  selector: 'app-root',
  template: `
    <div class="app-container">
      <header>
        <h1>{{ title }}</h1>
      </header>
      <main>
        <app-company-list></app-company-list>
        <app-company-form></app-company-form>
      </main>
    </div>
  `,
  standalone: true,
  imports: [CompanyListComponent, CompanyFormComponent],
  styles: [
    `
      .app-container {
        text-align: center;
        font-family: 'Poppins', Arial, sans-serif;
        background: linear-gradient(to bottom right, #fff8e1, #ffe082);
        min-height: 100vh;
        padding: 20px;
      }
      header {
        margin-bottom: 40px;
      }
      h1 {
        font-size: 48px;
        font-weight: 700;
        color: #ff6f00;
        text-transform: uppercase;
        text-shadow: 2px 2px #ffcc80;
        margin: 0;
      }
      main {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 40px;
      }
    `,
  ],
})
export class AppComponent {
  title = 'COMPANY PORTAL';
}
