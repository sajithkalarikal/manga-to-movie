import { createBrowserRouter } from 'react-router';
import { Root } from './Root';
import { Landing } from './pages/Landing';
import { Home } from './pages/Home';
import { Override } from './pages/Override';
import { Annotate } from './pages/Annotate';
import { Health } from './pages/Health';
import { NotFound } from './pages/NotFound';

export const router = createBrowserRouter([
  {
    path: '/ui_v2',
    Component: Root,
    children: [
      { 
        path: 'home', 
        Component: Home 
      },
      { 
        path: ':requestID/override', 
        Component: Override 
      },
      { 
        path: 'annotate', 
        Component: Annotate 
      },
      {
        path: 'health',
        Component: Health,
      },
      {
        index: true,
        Component: Landing,
      },
      {
        path: '*',
        Component: NotFound
      }
    ],
  },
  {
    path: '*',
    Component: () => {
      window.location.href = '/ui_v2';
      return null;
    }
  }
]);
